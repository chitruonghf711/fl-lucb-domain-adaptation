"""Federated learning server with Minibatch Cumulative Evaluation (Flower + PyTorch + SHAP).

The server runs FedAvg over clients, evaluates the aggregated model on a held-out test
set every round, and monitors client models using Minibatch Cumulative Accuracy Evaluation.

Minibatch Cumulative Evaluation Mechanism:
------------------------------------------
1. The server partitions the target domain test set into minibatches (stratified by class
   or sequential, configured by `minibatch_eval_size`).
2. When client evaluation is triggered (on accuracy plateau or forced every round), each
   client model is evaluated batch-by-batch up to `max_eval_minibatches`.
3. Cumulative accuracy is tracked after each batch:
   - If cumulative accuracy reaches `client_good_threshold_acc` (e.g. 0.85), the client is
     marked as 'GOOD' and evaluation halts immediately. This saves significant server compute
     and evaluation time (`samples_saved_pct`).
   - If all batches are evaluated and cumulative accuracy remains below the threshold, the
     client is marked as 'NEEDS_FEEDBACK'.
4. For clients needing feedback, the server generates adaptation guidance:
   - "abstract" protocol (default, privacy-preserving): Identifies the weak class and the
     top important features (via SHAP or gradient importance) WITHOUT sending any raw or
     masked test samples to the client.
   - "detailed" protocol (legacy): Sends masked test samples with top features kept and
     non-important features masked as NaN.
5. All minibatch evaluation records are logged to `minibatch_evaluation_log.csv`.
6. Feedback storage is refreshed each round to prevent stale feedback accumulation.

Usage:
    python server_tabular.py --minibatch-eval-size 300 --max-eval-minibatches 10 \
        --client-good-threshold-acc 0.85 --feedback-protocol abstract
"""

import argparse
import json
import logging
import os
import platform
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# === SERVER CONFIGURATION ===
CONFIG = {
    "server_address": "0.0.0.0:9000",
    "test_csv_path": "dataset/7030/test_server.csv",
    "target_column": "Exited",
    "num_rounds": 30,
    "min_clients": 1,
    "log_dir": "logs/exp_minibatch",
    "server_summary_filename": "summary_log.csv",
    "server_time_filename": "summary_time.csv",
    "server_config_filename": "server_config.json",
    "server_minibatch_filename": "minibatch_evaluation_log.csv",
    "plateau_window": 2,
    "plateau_delta": 0.01,
    "min_feedback_round": 1,
    "force_feedback_all_rounds": False,
    "shap_threshold": 0.05,
    "top_k_features": 10,
    "max_feedback_per_client": 1000,
    # Minibatch evaluation settings
    "minibatch_eval_size": 300,
    "max_eval_minibatches": 10,
    "client_good_threshold_acc": 0.85,
    "client_good_metric": "accuracy",
    "stratified_minibatch": True,
    "feedback_protocol": "abstract",  # "abstract" (privacy-preserving) or "detailed" (legacy)
    # Global budget ceiling settings
    "server_budget_fraction": 0.25,   # Global budget ceiling: max 25% of server total evaluation capacity
    "allow_budget_ceiling": True,     # True = enforce global budget ceiling E; False = unlimited
}

CONFIG["server_log_dir"] = os.path.join(CONFIG["log_dir"], "server")
CONFIG["server_feedback_dir"] = os.path.join(CONFIG["server_log_dir"], "feedback_logs")


# === DATA & MODEL ===
class SimpleNN(nn.Module):
    """Global model architecture; clients train an identical architecture."""

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().detach().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    model.load_state_dict({
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in zip(model.state_dict().keys(), parameters)
    })


def build_minibatches(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 300,
    stratified: bool = True,
    seed: int = 42
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Partition test features and labels into minibatches (stratified or sequential)."""
    n_samples = len(labels)
    if n_samples == 0:
        return []

    unique_classes = np.unique(labels)
    if stratified and len(unique_classes) > 1:
        rng = np.random.RandomState(seed)
        class_indices = {c: rng.permutation(np.where(labels == c)[0]) for c in unique_classes}

        n_batches = int(np.ceil(n_samples / max(1, batch_size)))
        batch_indices: List[List[int]] = [[] for _ in range(n_batches)]

        for c in unique_classes:
            c_idx = class_indices[c]
            splits = np.array_split(c_idx, n_batches)
            for b_idx, split in enumerate(splits):
                batch_indices[b_idx].extend(split.tolist())

        batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b_list in batch_indices:
            if not b_list:
                continue
            rng.shuffle(b_list)
            x_b = torch.tensor(features[b_list], dtype=torch.float32)
            y_b = torch.tensor(labels[b_list], dtype=torch.float32)
            batches.append((x_b, y_b))
        return batches
    else:
        batches = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            x_b = torch.tensor(features[start_idx:end_idx], dtype=torch.float32)
            y_b = torch.tensor(labels[start_idx:end_idx], dtype=torch.float32)
            batches.append((x_b, y_b))
        return batches


def load_test_data() -> Tuple[DataLoader, List[Tuple[torch.Tensor, torch.Tensor]], int, List[str], int]:
    """Load the test set, full DataLoader, stratified minibatches, and metadata."""
    test_df = pd.read_csv(CONFIG["test_csv_path"])
    features_df = test_df.drop(columns=[CONFIG["target_column"]])
    feature_names = features_df.columns.tolist()

    x_np = features_df.values.astype(np.float32)
    y_np = test_df[CONFIG["target_column"]].values.astype(np.float32)

    total_samples = len(y_np)
    testloader = DataLoader(
        TensorDataset(torch.tensor(x_np), torch.tensor(y_np)),
        batch_size=32,
        shuffle=False
    )

    minibatches = build_minibatches(
        features=x_np,
        labels=y_np,
        batch_size=CONFIG["minibatch_eval_size"],
        stratified=CONFIG["stratified_minibatch"]
    )

    return testloader, minibatches, x_np.shape[1], feature_names, total_samples


# === MINIBATCH EVALUATION ENGINE ===
def evaluate_model_minibatch(
    model: nn.Module,
    minibatches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float = 0.85,
    max_minibatches: int = 10,
    total_available_samples: Optional[int] = None,
    budget_remaining: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a client model over minibatches with cumulative accuracy, early stopping, and budget cap.

    Returns:
        batches_tested: Number of minibatches evaluated before stopping.
        samples_tested: Number of samples evaluated.
        cumulative_accuracy: Final cumulative accuracy over tested batches.
        status: "GOOD" if threshold met, "BUDGET_EXHAUSTED" if budget ran out, else "NEEDS_FEEDBACK".
        samples_saved_pct: Percentage of total available samples saved.
        misclassified_samples: List of (x_sample, true_label, pred_label).
        budget_exhausted: True if evaluation halted due to budget ceiling.
    """
    model.eval()
    if total_available_samples is None:
        total_available_samples = sum(len(b[1]) for b in minibatches)

    cumulative_correct = 0
    cumulative_tested = 0
    batches_tested = 0
    misclassified_samples: List[Tuple[torch.Tensor, int, int]] = []
    status = "NEEDS_FEEDBACK"
    cumulative_acc = 0.0
    budget_exhausted = False

    eval_batches = minibatches[:max_minibatches]
    for batch_x, batch_y in eval_batches:
        if budget_remaining is not None and budget_remaining <= 0:
            budget_exhausted = True
            break

        # If batch is larger than remaining budget, truncate batch to remaining budget
        if budget_remaining is not None and len(batch_y) > budget_remaining:
            if budget_remaining < 10:  # If fewer than 10 samples left, halt cleanly
                budget_exhausted = True
                break
            batch_x = batch_x[:budget_remaining]
            batch_y = batch_y[:budget_remaining]
            budget_exhausted = True

        batches_tested += 1
        with torch.no_grad():
            outputs = model(batch_x).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            targets = batch_y.int()
            batch_correct = (preds == targets).sum().item()

        cumulative_correct += batch_correct
        cumulative_tested += len(targets)
        if budget_remaining is not None:
            budget_remaining -= len(targets)

        for x_val, y_true, y_pred in zip(batch_x, targets, preds):
            if y_true.item() != y_pred.item():
                misclassified_samples.append((x_val, int(y_true.item()), int(y_pred.item())))

        cumulative_acc = cumulative_correct / max(1, cumulative_tested)

        if cumulative_acc >= threshold:
            status = "GOOD"
            break

        if budget_exhausted:
            break

    if status != "GOOD" and budget_exhausted and batches_tested < max_minibatches:
        status = "BUDGET_EXHAUSTED"

    samples_saved_pct = round(
        100.0 * max(0, total_available_samples - cumulative_tested) / max(1, total_available_samples), 2
    )

    return {
        "batches_tested": batches_tested,
        "samples_tested": cumulative_tested,
        "total_available_samples": total_available_samples,
        "cumulative_accuracy": round(cumulative_acc, 4),
        "threshold": threshold,
        "status": status,
        "samples_saved_pct": samples_saved_pct,
        "misclassified_samples": misclassified_samples,
        "budget_exhausted": budget_exhausted
    }


# === FEEDBACK GENERATION ===
def compute_shap_feedback(
    model: nn.Module,
    x_tensor: torch.Tensor,
    feature_names: List[str],
    threshold: float,
    top_k: int = 10
) -> List[Dict[str, float]]:
    """Legacy masked SHAP: retains values only for top-k features passing threshold."""
    if not HAS_SHAP:
        # Fallback if SHAP not installed: mock masking using feature variance
        x_numpy = x_tensor.detach().cpu().numpy()
        stds = np.std(x_numpy, axis=0)
        top_idx = set(np.argsort(stds)[::-1][:top_k])
        feedback_list = []
        for real_vec in x_numpy:
            feedback_list.append({
                name: float(real_val) if idx in top_idx else np.nan
                for idx, (name, real_val) in enumerate(zip(feature_names, real_vec))
            })
        return feedback_list

    model.eval()
    background = x_tensor[:100] if len(x_tensor) >= 100 else x_tensor
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x_tensor, check_additivity=False)
    shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
    x_numpy = x_tensor.detach().cpu().numpy()

    feedback_list = []
    for shap_vec, real_vec in zip(shap_values, x_numpy):
        abs_shap = np.abs(shap_vec)
        passed_idx = [i for i, val in enumerate(abs_shap) if val >= threshold]
        sorted_idx = sorted(passed_idx, key=lambda i: abs_shap[i], reverse=True)
        selected_idx = set(sorted_idx[:top_k])

        feedback_list.append({
            name: float(real_val) if idx in selected_idx else np.nan
            for idx, (name, real_val) in enumerate(zip(feature_names, real_vec))
        })
    return feedback_list


def compute_abstract_feedback(
    model: nn.Module,
    misclassified_samples: List[Tuple[torch.Tensor, int, int]],
    feature_names: List[str],
    top_k: int = 10
) -> Dict[str, Any]:
    """Privacy-preserving feedback: identifies weak class & top feature rankings without data leakage."""
    if not misclassified_samples:
        return {
            "protocol": "abstract",
            "weak_label": None,
            "weak_label_count": 0,
            "top_features": [],
            "feature_importance": {}
        }

    targets = [s[1] for s in misclassified_samples]
    most_common_label, count = Counter(targets).most_common(1)[0]
    filtered = [s for s in misclassified_samples if s[1] == most_common_label]

    if not filtered:
        return {
            "protocol": "abstract",
            "weak_label": int(most_common_label),
            "weak_label_count": count,
            "top_features": [],
            "feature_importance": {}
        }

    x_tensor = torch.stack([s[0] for s in filtered])
    feature_importance: Dict[str, float] = {}

    if HAS_SHAP and len(x_tensor) > 0:
        try:
            model.eval()
            background = x_tensor[:100] if len(x_tensor) >= 100 else x_tensor
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(x_tensor, check_additivity=False)
            shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            for name, score in zip(feature_names, mean_abs):
                feature_importance[name] = float(score)
        except Exception:
            mean_vals = np.mean(np.abs(x_tensor.detach().cpu().numpy()), axis=0)
            for name, score in zip(feature_names, mean_vals):
                feature_importance[name] = float(score)
    else:
        mean_vals = np.mean(np.abs(x_tensor.detach().cpu().numpy()), axis=0)
        for name, score in zip(feature_names, mean_vals):
            feature_importance[name] = float(score)

    sorted_features = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    top_feature_names = [k for k, _ in sorted_features[:top_k]]

    return {
        "protocol": "abstract",
        "weak_label": int(most_common_label),
        "weak_label_count": count,
        "top_features": top_feature_names,
        "feature_importance": {k: round(v, 4) for k, v in sorted_features[:top_k]}
    }


def compute_detailed_feedback(
    model: nn.Module,
    misclassified_samples: List[Tuple[torch.Tensor, int, int]],
    feature_names: List[str],
    target_column: str,
    threshold: float = 0.05,
    top_k: int = 10,
    limit: int = 1000
) -> Dict[str, Any]:
    """Legacy masked SHAP feedback with real feature values for top features."""
    if not misclassified_samples:
        return {"protocol": "detailed", "weak_label": None, "shap_feedback": [], "reference": []}

    targets = [s[1] for s in misclassified_samples]
    most_common_label, _ = Counter(targets).most_common(1)[0]
    filtered = [s for s in misclassified_samples if s[1] == most_common_label]

    if not filtered:
        return {"protocol": "detailed", "weak_label": int(most_common_label), "shap_feedback": [], "reference": []}

    x_tensor = torch.stack([s[0] for s in filtered])
    shap_feedbacks = compute_shap_feedback(model, x_tensor, feature_names, threshold, top_k=top_k)

    shap_entries = []
    real_samples = []
    for shap_feat, (x_val, label, _) in zip(shap_feedbacks, filtered):
        shap_entries.append({"features": shap_feat, "label": label})
        real_dict = {name: float(val.item()) for name, val in zip(feature_names, x_val)}
        real_dict[target_column] = label
        real_samples.append(real_dict)

    return {
        "protocol": "detailed",
        "weak_label": int(most_common_label),
        "shap_feedback": shap_entries[:limit],
        "reference": real_samples[:limit]
    }


# === STRATEGY ===
class MinibatchFeedbackFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with Minibatch Cumulative Accuracy Evaluation and Early Stopping."""

    def __init__(
        self,
        model: nn.Module,
        testloader: DataLoader,
        minibatches: List[Tuple[torch.Tensor, torch.Tensor]],
        feature_names: List[str],
        input_size: int,
        total_available_samples: int,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.model = model
        self.testloader = testloader
        self.minibatches = minibatches
        self.feature_names = feature_names
        self.input_size = input_size
        self.total_available_samples = total_available_samples

        self.client_id_map: Dict[str, str] = {}
        self.feedback_storage: Dict[str, Dict[str, Any]] = {}
        self.metrics_log: List[Dict[str, Any]] = []
        self.time_log: List[Dict[str, Any]] = []
        self.minibatch_logs: List[Dict[str, Any]] = []
        self.recent_accuracies: List[float] = []
        self.current_round = 0
        self.round_start = time.time()

    def evaluate_parameters(
        self, server_round: int, parameters: List[np.ndarray], log_result: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """Centralized evaluation on full test set for global tracking."""
        model = SimpleNN(self.input_size)
        set_parameters(model, parameters)
        model.eval()

        criterion = nn.BCEWithLogitsLoss()
        correct, total, loss_total = 0, 0, 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in self.testloader:
                outputs = model(inputs).squeeze()
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, targets)
                predicted = (torch.sigmoid(outputs) > 0.5).int()
                correct += (predicted == targets.int()).sum().item()
                total += len(targets)
                loss_total += loss.item() * len(targets)
                all_preds.extend(predicted.tolist())
                all_targets.extend(targets.int().tolist())

        acc = correct / max(1, total)
        avg_loss = loss_total / max(1, total)

        if log_result:
            tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
            self.recent_accuracies.append(acc)
            if len(self.recent_accuracies) > CONFIG["plateau_window"]:
                self.recent_accuracies.pop(0)

            self.metrics_log.append({
                "round": server_round,
                "accuracy": acc,
                "loss": avg_loss,
                "precision": precision_score(all_targets, all_preds, zero_division=0),
                "recall": recall_score(all_targets, all_preds, zero_division=0),
                "f1_score": f1_score(all_targets, all_preds, zero_division=0),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            })

        return avg_loss, {"accuracy": acc}

    def is_plateau(self, server_round: int) -> bool:
        """True once recent accuracies plateau within plateau_delta."""
        if server_round < CONFIG["min_feedback_round"]:
            return False
        if len(self.recent_accuracies) < CONFIG["plateau_window"]:
            return False
        window = CONFIG["plateau_window"]
        deltas = [
            abs(self.recent_accuracies[i + 1] - self.recent_accuracies[i])
            for i in range(-window, -1)
        ]
        return all(delta < CONFIG["plateau_delta"] for delta in deltas)

    def configure_fit(
        self, server_round: int, parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[Any, fl.common.FitIns]]:
        self.round_start = time.time()
        fit_ins_list = []
        for client_proxy in client_manager.sample(CONFIG["min_clients"]):
            client_id = self.client_id_map.get(client_proxy.cid, client_proxy.cid)
            feedback_bundle = self.feedback_storage.get(client_id, {})

            config = {
                "server_round": server_round,
                "feedback_protocol": feedback_bundle.get("protocol", CONFIG["feedback_protocol"]),
                "feedback": json.dumps(feedback_bundle),
                "reference": json.dumps(feedback_bundle.get("reference", []))
            }
            if feedback_bundle.get("protocol") == "detailed":
                config["legacy_feedback"] = json.dumps(feedback_bundle.get("shap_feedback", []))

            fit_ins_list.append((client_proxy, fl.common.FitIns(parameters, config)))
        return fit_ins_list

    def aggregate_fit(
        self, server_round: int, results: List[Any], failures: List[Any]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        self.current_round = server_round
        # Reset feedback storage so stale feedback from previous rounds is not reused!
        self.feedback_storage = {}

        start_agg = time.time()
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
        agg_time = time.time() - start_agg
        aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

        self.evaluate_parameters(server_round, aggregated_weights, log_result=True)

        fb_time = 0.0
        should_eval = self.is_plateau(server_round) or CONFIG.get("force_feedback_all_rounds", False)

        if should_eval:
            print(f"[SERVER] Plateau or forced evaluation active in round {server_round}. Running minibatch evaluation...")
            num_reporting_clients = len(results)
            total_server_capacity = num_reporting_clients * self.total_available_samples
            if CONFIG.get("allow_budget_ceiling", True):
                budget_fraction = CONFIG.get("server_budget_fraction", 0.25)
                global_budget_ceiling = int(budget_fraction * total_server_capacity)
            else:
                global_budget_ceiling = total_server_capacity

            total_server_samples_evaluated = 0
            total_server_eval_time = 0.0
            print(f"[SERVER] Global Budget Ceiling: {global_budget_ceiling} samples "
                  f"({CONFIG.get('server_budget_fraction', 0.25)*100:.1f}% of total capacity {total_server_capacity})")

            for client_proxy, fit_res in results:
                client_id = fit_res.metrics.get("client_id", client_proxy.cid)
                self.client_id_map[client_proxy.cid] = client_id

                client_model = SimpleNN(self.input_size)
                set_parameters(client_model, fl.common.parameters_to_ndarrays(fit_res.parameters))

                budget_remaining = max(0, global_budget_ceiling - total_server_samples_evaluated)
                if budget_remaining <= 0:
                    print(f"[SERVER] Global Budget Ceiling Exhausted! Halting evaluation for Client {client_id}.")
                    log_entry = {
                        "round": server_round,
                        "client_id": client_id,
                        "batches_tested": 0,
                        "samples_tested": 0,
                        "total_available_samples": self.total_available_samples,
                        "samples_saved_pct": 100.0,
                        "cumulative_accuracy": 0.0,
                        "threshold": CONFIG["client_good_threshold_acc"],
                        "status": "BUDGET_EXHAUSTED",
                        "eval_time_sec": 0.0,
                        "shap_time_sec": 0.0,
                        "feedback_samples_count": 0
                    }
                    self.minibatch_logs.append(log_entry)
                    continue

                eval_start = time.time()
                eval_res = evaluate_model_minibatch(
                    model=client_model,
                    minibatches=self.minibatches,
                    threshold=CONFIG["client_good_threshold_acc"],
                    max_minibatches=CONFIG["max_eval_minibatches"],
                    total_available_samples=self.total_available_samples,
                    budget_remaining=budget_remaining
                )
                eval_time = time.time() - eval_start
                total_server_samples_evaluated += eval_res["samples_tested"]
                total_server_eval_time += eval_time

                shap_time = 0.0
                feedback_count = 0

                if eval_res["status"] in ("NEEDS_FEEDBACK", "BUDGET_EXHAUSTED") and eval_res["misclassified_samples"]:
                    start_fb = time.time()
                    if CONFIG["feedback_protocol"] == "abstract":
                        feedback = compute_abstract_feedback(
                            model=client_model,
                            misclassified_samples=eval_res["misclassified_samples"],
                            feature_names=self.feature_names,
                            top_k=CONFIG["top_k_features"]
                        )
                        feedback_count = feedback.get("weak_label_count", len(eval_res["misclassified_samples"]))
                    else:
                        feedback = compute_detailed_feedback(
                            model=client_model,
                            misclassified_samples=eval_res["misclassified_samples"],
                            feature_names=self.feature_names,
                            target_column=CONFIG["target_column"],
                            threshold=CONFIG["shap_threshold"],
                            top_k=CONFIG["top_k_features"],
                            limit=CONFIG["max_feedback_per_client"]
                        )
                        feedback_count = len(feedback.get("shap_feedback", []))

                    shap_time = time.time() - start_fb
                    fb_time += shap_time
                    self.feedback_storage[client_id] = feedback

                    fb_file = os.path.join(
                        CONFIG["server_feedback_dir"],
                        f"feedback_round_{server_round}_client_{client_id}.json"
                    )
                    with open(fb_file, "w") as f:
                        json.dump(feedback, f, indent=2)
                else:
                    self.feedback_storage[client_id] = {}

                log_entry = {
                    "round": server_round,
                    "client_id": client_id,
                    "batches_tested": eval_res["batches_tested"],
                    "samples_tested": eval_res["samples_tested"],
                    "total_available_samples": eval_res["total_available_samples"],
                    "samples_saved_pct": eval_res["samples_saved_pct"],
                    "cumulative_accuracy": eval_res["cumulative_accuracy"],
                    "threshold": eval_res["threshold"],
                    "status": eval_res["status"],
                    "eval_time_sec": round(eval_time, 2),
                    "shap_time_sec": round(shap_time, 2),
                    "feedback_samples_count": feedback_count
                }
                self.minibatch_logs.append(log_entry)

                print(f"[SERVER] Client {client_id} Minibatch Eval: {eval_res['batches_tested']}/{CONFIG['max_eval_minibatches']} batches "
                      f"({eval_res['samples_tested']}/{eval_res['total_available_samples']} samples, saved {eval_res['samples_saved_pct']}%) "
                      f"Acc: {eval_res['cumulative_accuracy']:.4f} -> {eval_res['status']}")

            CONFIG["global_budget_ceiling"] = global_budget_ceiling
            CONFIG["total_server_samples_evaluated"] = total_server_samples_evaluated

        set_parameters(self.model, aggregated_weights)
        self.time_log.append({
            "round": server_round,
            "round_time_sec": round(time.time() - self.round_start, 2),
            "aggregate_time_sec": round(agg_time, 2),
            "eval_time_sec": round(total_server_eval_time, 2) if should_eval else 0.0,
            "feedback_time_sec": round(fb_time, 2),
            "samples_tested": total_server_samples_evaluated if should_eval else 0,
            "samples_saved_pct": round(100.0 * max(0, total_server_capacity - total_server_samples_evaluated) / max(1, total_server_capacity), 2) if should_eval else 0.0
        })

        self.save_reports(time.time() - self.round_start)
        return fl.common.ndarrays_to_parameters(aggregated_weights), {}

    def evaluate(
        self, server_round: int, parameters: fl.common.Parameters
    ) -> Tuple[float, Dict[str, float]]:
        return self.evaluate_parameters(
            server_round,
            fl.common.parameters_to_ndarrays(parameters),
            log_result=(server_round == 0)
        )

    def configure_evaluate(
        self, server_round: int, parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> None:
        return None

    def save_reports(self, total_time: float) -> None:
        log_dir = CONFIG["server_log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, CONFIG["server_summary_filename"])
        time_path = os.path.join(log_dir, CONFIG["server_time_filename"])
        minibatch_path = os.path.join(log_dir, CONFIG["server_minibatch_filename"])
        config_path = os.path.join(log_dir, CONFIG["server_config_filename"])

        if self.metrics_log:
            pd.DataFrame(self.metrics_log).to_csv(log_path, index=False)
        if self.time_log:
            pd.DataFrame(self.time_log).to_csv(time_path, index=False)
        if self.minibatch_logs:
            pd.DataFrame(self.minibatch_logs).to_csv(minibatch_path, index=False)

        CONFIG["total_time_sec"] = round(total_time, 2)
        with open(config_path, "w") as f:
            json.dump(CONFIG, f, indent=2)


def log_system_info() -> None:
    info = {
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False),
        "CPU Threads": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2)
    }
    print("[SERVER] System Info:")
    for key, value in info.items():
        print(f"[SERVER] {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-address", default=CONFIG["server_address"])
    parser.add_argument("--test-csv", default=CONFIG["test_csv_path"])
    parser.add_argument("--target-column", default=CONFIG["target_column"])
    parser.add_argument("--num-rounds", type=int, default=CONFIG["num_rounds"])
    parser.add_argument("--min-clients", type=int, default=CONFIG["min_clients"])
    parser.add_argument("--log-dir", default=CONFIG["log_dir"])
    parser.add_argument("--minibatch-eval-size", type=int, default=CONFIG["minibatch_eval_size"])
    parser.add_argument("--max-eval-minibatches", type=int, default=CONFIG["max_eval_minibatches"])
    parser.add_argument("--client-good-threshold-acc", type=float, default=CONFIG["client_good_threshold_acc"])
    parser.add_argument("--feedback-protocol", choices=["abstract", "detailed"], default=CONFIG["feedback_protocol"])
    parser.add_argument("--plateau-window", type=int, default=CONFIG["plateau_window"])
    parser.add_argument("--plateau-delta", type=float, default=CONFIG["plateau_delta"])
    parser.add_argument("--min-feedback-round", type=int, default=CONFIG["min_feedback_round"])
    parser.add_argument("--force-feedback-all-rounds", action="store_true", default=CONFIG["force_feedback_all_rounds"])
    parser.add_argument("--shap-threshold", type=float, default=CONFIG["shap_threshold"])
    parser.add_argument("--top-k-features", type=int, default=CONFIG["top_k_features"])
    parser.add_argument("--max-feedback-per-client", type=int, default=CONFIG["max_feedback_per_client"])
    parser.add_argument("--budget-fraction", type=float, default=CONFIG["server_budget_fraction"],
                        help="Global server budget ceiling as a fraction of total evaluation capacity (default: 0.25)")
    parser.add_argument("--no-budget-ceiling", action="store_true", default=False,
                        help="Disable global budget ceiling (evaluates without global sample cap)")
    return parser.parse_args()


# === MAIN ===
def main() -> None:
    args = parse_args()
    CONFIG["server_address"] = args.server_address
    CONFIG["test_csv_path"] = args.test_csv
    CONFIG["target_column"] = args.target_column
    CONFIG["num_rounds"] = args.num_rounds
    CONFIG["min_clients"] = args.min_clients
    CONFIG["log_dir"] = args.log_dir
    CONFIG["minibatch_eval_size"] = args.minibatch_eval_size
    CONFIG["max_eval_minibatches"] = args.max_eval_minibatches
    CONFIG["client_good_threshold_acc"] = args.client_good_threshold_acc
    CONFIG["feedback_protocol"] = args.feedback_protocol
    CONFIG["plateau_window"] = args.plateau_window
    CONFIG["plateau_delta"] = args.plateau_delta
    CONFIG["min_feedback_round"] = args.min_feedback_round
    CONFIG["force_feedback_all_rounds"] = args.force_feedback_all_rounds
    CONFIG["shap_threshold"] = args.shap_threshold
    CONFIG["top_k_features"] = args.top_k_features
    CONFIG["max_feedback_per_client"] = args.max_feedback_per_client

    if args.no_budget_ceiling:
        CONFIG["allow_budget_ceiling"] = False
        CONFIG["server_budget_fraction"] = 1.0
    else:
        CONFIG["allow_budget_ceiling"] = True
        CONFIG["server_budget_fraction"] = args.budget_fraction

    CONFIG["server_log_dir"] = os.path.join(CONFIG["log_dir"], "server")
    CONFIG["server_feedback_dir"] = os.path.join(CONFIG["server_log_dir"], "feedback_logs")

    start_time = time.time()
    os.makedirs(CONFIG["server_log_dir"], exist_ok=True)
    os.makedirs(CONFIG["server_feedback_dir"], exist_ok=True)

    print(f"[SERVER] Logging to {CONFIG['server_log_dir']}")
    print(f"[SERVER] Feedback Protocol: {CONFIG['feedback_protocol'].upper()}")
    print(f"[SERVER] Minibatch Config: batch_size={CONFIG['minibatch_eval_size']}, max_batches={CONFIG['max_eval_minibatches']}, threshold={CONFIG['client_good_threshold_acc']}")
    print("[SERVER] Loading test dataset...")
    testloader, minibatches, input_size, feature_names, total_samples = load_test_data()

    log_system_info()
    print("[SERVER] Server training started.")

    model = SimpleNN(input_size)
    strategy = MinibatchFeedbackFedAvg(
        model=model,
        testloader=testloader,
        minibatches=minibatches,
        feature_names=feature_names,
        input_size=input_size,
        total_available_samples=total_samples,
        min_fit_clients=CONFIG["min_clients"],
        min_available_clients=CONFIG["min_clients"],
        min_evaluate_clients=1,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model))
    )

    print("[SERVER] Starting Flower server...")
    fl.server.start_server(
        server_address=CONFIG["server_address"],
        config=fl.server.ServerConfig(num_rounds=CONFIG["num_rounds"]),
        strategy=strategy
    )

    strategy.save_reports(time.time() - start_time)


if __name__ == "__main__":
    main()
