"""Federated tabular server with budgeted LUCB feedback selection.

The default abstract protocol sends only weak labels and aggregate SHAP feature
ranks. It never sends target-domain feature values. The historical full scan
and raw payload behavior requires explicit compatibility flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import psutil
import shap
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl_lucb import (  # noqa: E402
    BatchScore,
    LUCBSelector,
    MinibatchPlan,
    SelectionConfig,
    build_audit_record,
    sanitize_audit_record,
    validate_feedback_payload,
    write_audit_json,
)
from fl_lucb.models import SelectionState  # noqa: E402


DEFAULT_CONFIG: Dict[str, Any] = {
    "server_address": "0.0.0.0:9000",
    "test_csv_path": "dataset/7030/test_server.csv",
    "target_column": "Exited",
    "max_feedback_per_client": 1000,
    "num_rounds": 30,
    "min_clients": 1,
    "log_dir": "logs/lucb_pilot",
    "server_summary_filename": "summary_log.csv",
    "server_time_filename": "summary_time.csv",
    "server_config_filename": "server_config.json",
    "global_monitoring_filename": "global_monitoring.jsonl",
    "plateau_window": 3,
    "plateau_delta": 0.01,
    "min_feedback_round": 3,
    "shap_threshold": 0.05,
    "top_k_features": 10,
    "selection_mode": "lucb",
    "lucb_budget_fraction": 0.25,
    "lucb_batch_size": 64,
    "lucb_top_k": None,
    "lucb_epsilon": 0.02,
    "lucb_delta": 0.05,
    "weak_class_accuracy": 0.85,
    "seed": 42,
    "lucb_prioritization": "strict",
    "priority_decay": 0.5,
    "feedback_protocol": "abstract",
    "allow_raw_feedback": False,
    "test_force_feedback": False,
}
CONFIG: Dict[str, Any] = dict(DEFAULT_CONFIG)


def refresh_run_paths(config: Dict[str, Any]) -> None:
    """Derive run-local paths without touching historical result directories."""

    root = os.path.join(config["log_dir"], "server")
    config["server_log_dir"] = root
    config["server_feedback_dir"] = os.path.join(root, "feedback_logs")
    config["server_audit_dir"] = os.path.join(root, "audit")
    config["server_monitoring_path"] = os.path.join(root, config["global_monitoring_filename"])


refresh_run_paths(CONFIG)


def _json_default(value: Any) -> Any:
    """Serialize numerical scalars safely and never serialize tensor contents."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape)}
    return str(value)


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Persist a JSON record with stable formatting."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def append_jsonl(path: str, payload: Mapping[str, Any]) -> None:
    """Append a monitoring record independent of the LUCB budget."""

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        handle.write("\n")


def selection_config_from_runtime(config: Mapping[str, Any]) -> SelectionConfig:
    """Create the shared selector configuration from the runtime settings."""

    selection_config = SelectionConfig(
        selection_mode=config["selection_mode"],
        lucb_budget_fraction=float(config["lucb_budget_fraction"]),
        lucb_batch_size=int(config["lucb_batch_size"]),
        lucb_top_k=config["lucb_top_k"],
        lucb_epsilon=float(config["lucb_epsilon"]),
        lucb_delta=float(config["lucb_delta"]),
        weak_class_accuracy=float(config["weak_class_accuracy"]),
        seed=int(config["seed"]),
        lucb_prioritization=config["lucb_prioritization"],
        priority_decay=float(config["priority_decay"]),
    )
    selection_config.validate()
    return selection_config


def validate_runtime_config(config: Mapping[str, Any]) -> None:
    """Reject invalid or unsafe combinations before starting Flower."""

    selection_config_from_runtime(config)
    if config["feedback_protocol"] not in {"abstract", "legacy"}:
        raise ValueError("feedback_protocol must be 'abstract' or 'legacy'")
    if config["feedback_protocol"] == "legacy" and not config["allow_raw_feedback"]:
        raise ValueError("feedback_protocol='legacy' requires --allow-raw-feedback")
    if int(config["max_feedback_per_client"]) <= 0:
        raise ValueError("max_feedback_per_client must be positive")


def audit_configuration() -> Dict[str, Any]:
    """Return effective safe settings that make a feedback event reproducible."""

    configuration = selection_config_from_runtime(CONFIG).to_dict()
    configuration.update({
        "feedback_protocol": CONFIG["feedback_protocol"],
        "legacy_opt_in": bool(CONFIG["allow_raw_feedback"]),
        "max_feedback_per_client": int(CONFIG["max_feedback_per_client"]),
        "shap_threshold": float(CONFIG["shap_threshold"]),
        "top_k_features": int(CONFIG["top_k_features"]),
        "test_force_feedback": bool(CONFIG["test_force_feedback"]),
    })
    return configuration


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse server and shared LUCB configuration."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-address", default=CONFIG["server_address"], help="Flower bind address")
    parser.add_argument("--test-csv", default=CONFIG["test_csv_path"], help="Target-domain test CSV")
    parser.add_argument("--log-dir", default=CONFIG["log_dir"], help="New run output directory")
    parser.add_argument("--num-rounds", type=int, default=CONFIG["num_rounds"], help="Training rounds")
    parser.add_argument("--min-clients", type=int, default=CONFIG["min_clients"], help="Clients per round")
    parser.add_argument("--selection-mode", choices=("lucb", "legacy_full_scan"),
                        default=CONFIG["selection_mode"], help="Budgeted LUCB or explicit baseline")
    parser.add_argument("--lucb-budget-fraction", type=float, default=CONFIG["lucb_budget_fraction"],
                        help="Fraction of full client-by-sample work available to LUCB")
    parser.add_argument("--lucb-batch-size", type=int, default=CONFIG["lucb_batch_size"],
                        help="Deterministic LUCB minibatch size")
    parser.add_argument("--lucb-top-k", type=int, default=CONFIG["lucb_top_k"],
                        help="Weak clients to retain; omit for an automatic cap of three")
    parser.add_argument("--lucb-epsilon", type=float, default=CONFIG["lucb_epsilon"],
                        help="LUCB boundary tolerance")
    parser.add_argument("--lucb-delta", type=float, default=CONFIG["lucb_delta"],
                        help="Wilson confidence failure probability")
    parser.add_argument("--weak-class-accuracy", type=float, default=CONFIG["weak_class_accuracy"],
                        help="A class is weak only below this Wilson upper bound")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="Minibatch plan seed")
    parser.add_argument("--lucb-prioritization", choices=("strict", "decay_priority"),
                        default=CONFIG["lucb_prioritization"],
                        help="Strict resets evidence; decay_priority only reuses scheduling priority")
    parser.add_argument("--priority-decay", type=float, default=CONFIG["priority_decay"],
                        help="Experimental priority decay for decay_priority")
    parser.add_argument("--feedback-protocol", choices=("abstract", "legacy"),
                        default=CONFIG["feedback_protocol"], help="Abstract feedback is the default")
    parser.add_argument("--allow-raw-feedback", action="store_true",
                        help="Required acknowledgement for legacy raw feedback")
    parser.add_argument("--max-feedback-per-client", type=int,
                        default=CONFIG["max_feedback_per_client"], help="Maximum explained errors")
    parser.add_argument("--test-force-feedback", action="store_true",
                        help="Test-only: bypass the global plateau trigger")
    return parser.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    """Apply CLI values to the single server runtime configuration."""

    CONFIG.update({
        "server_address": args.server_address,
        "test_csv_path": args.test_csv,
        "log_dir": args.log_dir,
        "num_rounds": args.num_rounds,
        "min_clients": args.min_clients,
        "selection_mode": args.selection_mode,
        "lucb_budget_fraction": args.lucb_budget_fraction,
        "lucb_batch_size": args.lucb_batch_size,
        "lucb_top_k": args.lucb_top_k,
        "lucb_epsilon": args.lucb_epsilon,
        "lucb_delta": args.lucb_delta,
        "weak_class_accuracy": args.weak_class_accuracy,
        "seed": args.seed,
        "lucb_prioritization": args.lucb_prioritization,
        "priority_decay": args.priority_decay,
        "feedback_protocol": args.feedback_protocol,
        "allow_raw_feedback": bool(args.allow_raw_feedback),
        "max_feedback_per_client": args.max_feedback_per_client,
        "test_force_feedback": bool(args.test_force_feedback),
    })
    refresh_run_paths(CONFIG)
    validate_runtime_config(CONFIG)


def load_test_data() -> Tuple[DataLoader, int, List[str]]:
    """Load target data in fixed row order, which supplies stable sample IDs."""

    test_df = pd.read_csv(CONFIG["test_csv_path"])
    features = test_df.drop(columns=[CONFIG["target_column"]])
    x_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(test_df[CONFIG["target_column"]].values, dtype=torch.float32)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=32, shuffle=False), x_tensor.shape[1], features.columns.tolist()


class SimpleNN(nn.Module):
    """Shared tabular MLP architecture."""

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [value.cpu().detach().numpy() for value in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: Sequence[np.ndarray]) -> None:
    model.load_state_dict({
        key: torch.tensor(value, dtype=torch.float32)
        for key, value in zip(model.state_dict().keys(), parameters)
    })


def model_collection_fingerprint(client_models: Mapping[str, nn.Module]) -> str:
    """Fingerprint precisely the client models used as selector arms."""

    digest = hashlib.sha256()
    for client_id in sorted(client_models):
        digest.update(client_id.encode("utf-8"))
        for value in client_models[client_id].state_dict().values():
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _coerce_shap_matrix(values: Any, feature_count: int) -> np.ndarray:
    """Normalize DeepExplainer output to samples by features."""

    matrix = np.asarray(values[0] if isinstance(values, list) else values)
    if matrix.ndim == 3 and matrix.shape[-1] == 1:
        matrix = matrix[..., 0]
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != feature_count:
        raise ValueError("Unexpected SHAP output shape for tabular feedback")
    return matrix


def compute_abstract_feature_ranks(
    model: nn.Module, x_tensor: torch.Tensor, feature_names: Sequence[str], threshold: float, top_k: int
) -> List[Dict[str, Any]]:
    """Return aggregate SHAP ranks; no target-domain feature values leave the server."""

    if not len(x_tensor):
        return []
    model.eval()
    explainer = shap.DeepExplainer(model, x_tensor[: min(100, len(x_tensor))])
    matrix = _coerce_shap_matrix(explainer.shap_values(x_tensor, check_additivity=False), len(feature_names))
    importances = np.mean(np.abs(matrix), axis=0)
    ranked = sorted(range(len(feature_names)), key=lambda index: float(importances[index]), reverse=True)
    selected = [index for index in ranked if float(importances[index]) >= threshold][:top_k]
    if not selected:
        selected = ranked[:top_k]
    return [
        {"feature": str(feature_names[index]), "rank": rank,
         "importance": round(float(importances[index]), 8)}
        for rank, index in enumerate(selected, start=1)
    ]


def compute_legacy_shap_feedback(
    model: nn.Module, x_tensor: torch.Tensor, feature_names: Sequence[str], threshold: float, top_k: int
) -> List[Dict[str, float]]:
    """Build the historical masked-value payload for explicit legacy mode only."""

    model.eval()
    explainer = shap.DeepExplainer(model, x_tensor[: min(100, len(x_tensor))])
    matrix = _coerce_shap_matrix(explainer.shap_values(x_tensor, check_additivity=False), len(feature_names))
    feedback = []
    for shap_vector, raw_values in zip(matrix, x_tensor.detach().cpu().numpy()):
        importance = np.abs(shap_vector)
        ranked = sorted(range(len(importance)), key=lambda index: float(importance[index]), reverse=True)
        selected = [index for index in ranked if float(importance[index]) >= threshold][:top_k]
        feedback.append({
            name: float(value) if index in selected else float("nan")
            for index, (name, value) in enumerate(zip(feature_names, raw_values))
        })
    return feedback


def label_to_json(value: Any) -> Any:
    """Convert scalar labels and sample IDs to JSON-compatible values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def stable_client_id(value: Any) -> Optional[str]:
    """Return a usable client-reported ID or ``None`` for a session fallback."""

    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_abstract_tabular_payload(
    client_id: str,
    model: nn.Module,
    errors_by_label: Mapping[Any, Sequence[torch.Tensor]],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Construct an abstract payload only from already-scored selected errors."""

    weak_labels = [label_to_json(label) for label in sorted(errors_by_label, key=repr)]
    samples = [
        sample
        for label in sorted(errors_by_label, key=repr)
        for sample in errors_by_label[label]
    ][: int(CONFIG["max_feedback_per_client"])]
    ranks = compute_abstract_feature_ranks(
        model, torch.stack(samples), feature_names, float(CONFIG["shap_threshold"]), int(CONFIG["top_k_features"])
    ) if samples else []
    return validate_feedback_payload(
        {
            "protocol": "abstract", "domain": "tabular", "client_id": client_id,
            "weak_labels": weak_labels,
            "feature_ranking": ranks,
        },
        protocol="abstract", allow_raw_feedback=False, domain="tabular"
    )


def build_legacy_tabular_payload(
    model: nn.Module, errors_by_label: Mapping[Any, Sequence[torch.Tensor]], feature_names: Sequence[str]
) -> Dict[str, Any]:
    """Build raw historical feedback after the operator opted into legacy mode."""

    candidates = [(sample, label) for label in sorted(errors_by_label, key=repr) for sample in errors_by_label[label]]
    if not candidates:
        return {"shap_feedback": [], "reference": []}
    majority_label, _ = Counter(label for _, label in candidates).most_common(1)[0]
    selected = [(sample, label) for sample, label in candidates if label == majority_label]
    selected = selected[: int(CONFIG["max_feedback_per_client"])]
    x_tensor = torch.stack([sample for sample, _ in selected])
    masked = compute_legacy_shap_feedback(
        model, x_tensor, feature_names, float(CONFIG["shap_threshold"]), int(CONFIG["top_k_features"])
    )
    reference = []
    for sample, label in selected:
        row = {name: float(value.item()) for name, value in zip(feature_names, sample)}
        row[CONFIG["target_column"]] = label_to_json(label)
        reference.append(row)
    return {
        "protocol": "legacy",
        "domain": "tabular",
        "weak_labels": [label_to_json(majority_label)],
        "shap_feedback": [
            {"features": values, "label": label_to_json(label)}
            for values, (_, label) in zip(masked, selected)
        ],
        "reference": reference,
    }


class FeedbackFedAvg(fl.server.strategy.FedAvg):
    """FedAvg with separate global monitoring and feedback selection."""

    def __init__(self, model: nn.Module, testloader: DataLoader, feature_names: List[str], input_size: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model
        self.testloader = testloader
        self.feature_names = feature_names
        self.input_size = input_size
        if not isinstance(testloader.dataset, TensorDataset):
            raise TypeError("Tabular LUCB requires TensorDataset target data")
        self.test_inputs, self.test_targets = testloader.dataset.tensors
        self.sample_index_by_id = {str(index): index for index in range(len(self.test_targets))}
        self.client_id_map: Dict[str, str] = {}
        self.feedback_storage: Dict[str, Dict[str, Any]] = {}
        self.metrics_log: List[Dict[str, Any]] = []
        self.time_log: List[Dict[str, Any]] = []
        self.recent_accuracies: List[float] = []
        self.current_round = 0
        self.round_start = time.time()
        self.lucb_state = SelectionState()
        self._last_lucb_fingerprint: Optional[str] = None
        self._selection_observations: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._last_selection_summary: Dict[str, Any] = {}

    def evaluate_parameters(self, server_round: int, parameters: Sequence[np.ndarray], log_result: bool = True) -> Tuple[float, Dict[str, float]]:
        """Evaluate global weights; this full monitoring scan is excluded from M."""

        model = SimpleNN(self.input_size)
        set_parameters(model, parameters)
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        correct = total = 0
        loss_total = 0.0
        predictions: List[int] = []
        targets_all: List[int] = []
        with torch.no_grad():
            for inputs, targets in self.testloader:
                logits = model(inputs).squeeze()
                loss = criterion(logits, targets)
                batch_predictions = (torch.sigmoid(logits) > 0.5).int()
                correct += (batch_predictions == targets.int()).sum().item()
                total += len(targets)
                loss_total += loss.item() * len(targets)
                predictions.extend(int(value) for value in batch_predictions.tolist())
                targets_all.extend(int(value) for value in targets.int().tolist())
        accuracy = correct / total if total else 0.0
        average_loss = loss_total / total if total else 0.0
        if log_result:
            tn, fp, fn, tp = confusion_matrix(targets_all, predictions, labels=[0, 1]).ravel()
            record = {
                "round": server_round, "accuracy": accuracy, "loss": average_loss,
                "precision": precision_score(targets_all, predictions, zero_division=0),
                "recall": recall_score(targets_all, predictions, zero_division=0),
                "f1_score": f1_score(targets_all, predictions, zero_division=0),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            }
            self.metrics_log.append(record)
            self.recent_accuracies.append(accuracy)
            if len(self.recent_accuracies) > int(CONFIG["plateau_window"]):
                self.recent_accuracies.pop(0)
            append_jsonl(CONFIG["server_monitoring_path"], {
                "scope": "global_model_monitoring",
                "model_sample_evaluations_excluded_from_lucb": total,
                **record,
            })
        return average_loss, {"accuracy": accuracy}

    def is_plateau(self, server_round: int) -> bool:
        """Detect a plateau solely from independently logged global monitoring."""

        if server_round < int(CONFIG["min_feedback_round"]):
            return False
        window = int(CONFIG["plateau_window"])
        if len(self.recent_accuracies) < window:
            return False
        return all(
            abs(self.recent_accuracies[index + 1] - self.recent_accuracies[index]) < float(CONFIG["plateau_delta"])
            for index in range(-window, -1)
        )

    @staticmethod
    def _sample_key(sample_id: Any) -> str:
        return str(sample_id)

    def _make_plans(self, client_ids: Iterable[str]) -> Dict[str, MinibatchPlan]:
        labels = [int(value) for value in self.test_targets.int().tolist()]
        sample_ids = list(range(len(labels)))
        return {
            client_id: MinibatchPlan.build(
                client_id=client_id, labels=labels, batch_size=int(CONFIG["lucb_batch_size"]),
                seed=int(CONFIG["seed"]), sample_ids=sample_ids
            )
            for client_id in sorted(client_ids)
        }

    def _score_planned_batch(self, client_models: Mapping[str, nn.Module], client_id: str, batch: Any) -> BatchScore:
        """Evaluate exactly the batch requested by the selector and nothing else."""

        sample_ids = tuple(batch.sample_ids)
        positions = [self.sample_index_by_id[self._sample_key(sample_id)] for sample_id in sample_ids]
        model = client_models[client_id]
        model.eval()
        with torch.no_grad():
            predictions = (torch.sigmoid(model(self.test_inputs[positions]).squeeze()) > 0.5).int().tolist()
        labels = [int(value) for value in batch.labels]
        predictions = [int(value) for value in predictions]
        score = BatchScore.from_predictions(labels, predictions)
        observations = self._selection_observations.setdefault(client_id, {})
        for sample_id, label, prediction, correct in zip(sample_ids, labels, predictions, score.correctness):
            observations[self._sample_key(sample_id)] = {
                "label": label, "prediction": prediction, "correct": bool(correct),
            }
        return score

    def _errors_for_selected_classes(
        self, client_id: str, labels: Iterable[Any], scored_sample_ids: Iterable[Any]
    ) -> Dict[Any, List[torch.Tensor]]:
        """Return only sampled errors belonging to selected weak classes."""

        selected_labels = {label_to_json(label) for label in labels}
        errors: Dict[Any, List[torch.Tensor]] = defaultdict(list)
        for sample_id in scored_sample_ids:
            observation = self._selection_observations.get(client_id, {}).get(self._sample_key(sample_id))
            if not observation or observation["correct"] or observation["label"] not in selected_labels:
                continue
            errors[observation["label"]].append(self.test_inputs[self.sample_index_by_id[self._sample_key(sample_id)]])
        return errors

    def _full_scan_errors(self, model: nn.Module) -> Dict[Any, List[torch.Tensor]]:
        """Compatibility-only full target scan for the explicit baseline."""

        errors: Dict[Any, List[torch.Tensor]] = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for inputs, targets in self.testloader:
                predictions = (torch.sigmoid(model(inputs).squeeze()) > 0.5).int()
                for sample, label, prediction in zip(inputs, targets.int(), predictions):
                    if int(label.item()) != int(prediction.item()):
                        errors[int(label.item())].append(sample)
        return errors

    def _store_feedback(
        self, client_id: str, model: nn.Module, errors_by_label: Mapping[Any, Sequence[torch.Tensor]]
    ) -> float:
        """Explain caller-provided error rows and retain a protocol-specific bundle."""

        started = time.time()
        if CONFIG["feedback_protocol"] == "abstract":
            payload = build_abstract_tabular_payload(client_id, model, errors_by_label, self.feature_names)
            bundle = {"feedback": payload}
        else:
            raw_payload = validate_feedback_payload(
                build_legacy_tabular_payload(model, errors_by_label, self.feature_names),
                protocol="legacy", allow_raw_feedback=True, domain="tabular"
            )
            bundle = {
                "feedback": raw_payload.get("shap_feedback", []),
                "reference": raw_payload.get("reference", []),
            }
        self.feedback_storage[client_id] = bundle
        write_json(
            os.path.join(CONFIG["server_feedback_dir"], f"feedback_round_{self.current_round}_client_{client_id}.json"),
            bundle,
        )
        return time.time() - started

    def _write_audit(self, audit: Mapping[str, Any]) -> None:
        """Persist a sanitized event audit that cannot contain target data."""

        safe = sanitize_audit_record(audit, protocol=CONFIG["feedback_protocol"])
        write_audit_json(
            os.path.join(CONFIG["server_audit_dir"], f"feedback_round_{self.current_round}.json"), safe
        )

    def _record_duplicate_client_ids(self, duplicate_ids: Sequence[str]) -> None:
        """Persist a routing guard event without selecting or exposing any client feedback."""

        self.feedback_storage.clear()
        duplicate_values = sorted(set(duplicate_ids))
        self._last_selection_summary = {
            "selection_mode": CONFIG["selection_mode"],
            "status": "skipped",
            "stop_reason": "duplicate_stable_client_id",
            "duplicate_client_ids": duplicate_values,
        }
        self._write_audit({
            "event_type": "feedback_routing_guard",
            "protocol": CONFIG["feedback_protocol"],
            "configuration": audit_configuration(),
            "round": self.current_round,
            "status": "skipped",
            "stop_reason": "duplicate_stable_client_id",
            "duplicate_client_ids": duplicate_values,
        })
        warnings.warn(
            "Skipping tabular feedback because multiple fit results reported the same stable client_id: "
            f"{duplicate_values}.",
            RuntimeWarning,
        )

    def _run_lucb_feedback(self, client_models: Mapping[str, nn.Module]) -> Tuple[float, Dict[str, Any]]:
        """Run bounded selection and explain no rows beyond its observed sample IDs."""

        self._selection_observations = {}
        fingerprint = model_collection_fingerprint(client_models)
        if CONFIG["lucb_prioritization"] == "strict" and self._last_lucb_fingerprint not in {None, fingerprint}:
            self.lucb_state = SelectionState()
        result = LUCBSelector(selection_config_from_runtime(CONFIG)).select(
            self._make_plans(client_models),
            lambda client_id, batch: self._score_planned_batch(client_models, client_id, batch),
            model_fingerprint=fingerprint,
            state=self.lucb_state,
        )
        self._last_lucb_fingerprint = fingerprint
        feedback_seconds = 0.0
        feedback_clients: List[str] = []
        if result.feedback_ready:
            for client_id in result.selected_client_ids:
                assessments = result.selected_classes.get(client_id, ())
                errors = self._errors_for_selected_classes(
                    client_id,
                    (assessment.label for assessment in assessments if assessment.status == "weak"),
                    result.scored_sample_ids.get(client_id, ()),
                )
                if errors:
                    feedback_seconds += self._store_feedback(client_id, client_models[client_id], errors)
                    feedback_clients.append(client_id)
        audit = build_audit_record(
            event_id=f"tabular-round-{self.current_round}",
            configuration=audit_configuration(),
            model_fingerprint=result.model_fingerprint,
            status=result.status,
            stop_reason=result.stop_reason,
            model_sample_evaluations=result.model_sample_evaluations,
            budget=result.budget,
            selected_client_ids=result.selected_client_ids,
            selected_classes=result.selected_classes,
            confidence_bounds=result.confidence_bounds,
            coverage=result.coverage,
            scored_batches=result.scored_batches,
            boundary_gap=result.boundary_gap,
            explanation_cost={"explained_clients": len(feedback_clients), "seconds": round(feedback_seconds, 6)},
            protocol=CONFIG["feedback_protocol"],
            metadata={
                "domain": "tabular",
                "feedback_protocol": CONFIG["feedback_protocol"],
                "feedback_client_ids": feedback_clients,
            },
        )
        self._write_audit(audit)
        return feedback_seconds, {
            "selection_mode": "lucb", "status": result.status, "stop_reason": result.stop_reason,
            "model_sample_evaluations": result.model_sample_evaluations, "budget": result.budget,
            "boundary_gap": result.boundary_gap, "selected_client_ids": list(result.selected_client_ids),
            "feedback_client_ids": feedback_clients,
        }

    def _run_legacy_full_scan_feedback(self, client_models: Mapping[str, nn.Module]) -> Tuple[float, Dict[str, Any]]:
        """Run the original full scan only when selection_mode explicitly requests it."""

        elapsed = 0.0
        feedback_clients = []
        for client_id, model in client_models.items():
            errors = self._full_scan_errors(model)
            if errors:
                elapsed += self._store_feedback(client_id, model, errors)
                feedback_clients.append(client_id)
            else:
                self.feedback_storage[client_id] = {"feedback": []}
        scanned = len(client_models) * len(self.test_targets)
        audit = build_audit_record(
            event_id=f"tabular-legacy-round-{self.current_round}",
            configuration=audit_configuration(),
            model_fingerprint=model_collection_fingerprint(client_models),
            status="ok",
            stop_reason="explicit_legacy_full_scan",
            model_sample_evaluations=scanned,
            budget=scanned,
            selected_client_ids=sorted(client_models),
            selected_classes={client_id: () for client_id in sorted(client_models)},
            confidence_bounds={},
            coverage={
                client_id: {"scored_sample_count": len(self.test_targets), "mode": "full_scan"}
                for client_id in sorted(client_models)
            },
            scored_batches={},
            boundary_gap=None,
            explanation_cost={
                "explained_clients": len(feedback_clients),
                "seconds": round(elapsed, 6),
            },
            protocol=CONFIG["feedback_protocol"],
            domain="tabular",
            feedback_protocol=CONFIG["feedback_protocol"],
            metadata={
                "selection_mode": "legacy_full_scan",
                "feedback_client_ids": feedback_clients,
            },
        )
        self._write_audit(audit)
        return elapsed, {
            "selection_mode": "legacy_full_scan", "status": "ok", "stop_reason": "explicit_legacy_full_scan",
            "model_sample_evaluations": scanned, "budget": scanned, "boundary_gap": None,
            "selected_client_ids": sorted(client_models), "feedback_client_ids": feedback_clients,
        }

    def configure_fit(self, server_round: int, parameters: fl.common.Parameters,
                      client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[Any, fl.common.FitIns]]:
        self.round_start = time.time()
        instructions = []
        routed_client_ids: Dict[str, str] = {}
        for client_proxy in client_manager.sample(CONFIG["min_clients"]):
            client_id = self.client_id_map.get(client_proxy.cid, client_proxy.cid)
            duplicate_route = client_id in routed_client_ids
            if duplicate_route:
                warnings.warn(
                    "Withholding tabular feedback from a duplicate stable client_id route: "
                    f"{client_id} (proxies {routed_client_ids[client_id]} and {client_proxy.cid}).",
                    RuntimeWarning,
                )
                bundle = {"feedback": []}
            else:
                routed_client_ids[client_id] = client_proxy.cid
                bundle = self.feedback_storage.get(client_id, {"feedback": []})
            config: Dict[str, Any] = {
                "feedback": json.dumps(bundle["feedback"], default=_json_default),
                "feedback_protocol": CONFIG["feedback_protocol"], "server_round": server_round,
            }
            if CONFIG["feedback_protocol"] == "legacy":
                config["reference"] = json.dumps(bundle.get("reference", []), default=_json_default)
            instructions.append((client_proxy, fl.common.FitIns(parameters, config)))
        return instructions

    def aggregate_fit(self, server_round: int, results: List[Any], failures: List[Any]) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        self.current_round = server_round
        started = time.time()
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None, {}
        aggregation_seconds = time.time() - started
        weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
        self.evaluate_parameters(server_round, weights, log_result=True)
        client_models: Dict[str, nn.Module] = {}
        duplicate_client_ids: List[str] = []
        for client_proxy, fit_result in results:
            reported_id = stable_client_id(fit_result.metrics.get("client_id"))
            client_id = reported_id or f"proxy-{client_proxy.cid}"
            if reported_id is None:
                warnings.warn(
                    f"Client proxy {client_proxy.cid} did not report a valid stable client_id; "
                    "using a session-scoped fallback for this feedback event.",
                    RuntimeWarning,
                )
            self.client_id_map[client_proxy.cid] = client_id
            if client_id in client_models:
                duplicate_client_ids.append(client_id)
                continue
            model = SimpleNN(self.input_size)
            set_parameters(model, fl.common.parameters_to_ndarrays(fit_result.parameters))
            client_models[client_id] = model
        feedback_seconds = 0.0
        self._last_selection_summary = {}
        triggered = bool(CONFIG["test_force_feedback"]) or self.is_plateau(server_round)
        if triggered and client_models:
            if duplicate_client_ids:
                self._record_duplicate_client_ids(duplicate_client_ids)
            elif CONFIG["selection_mode"] == "lucb":
                feedback_seconds, self._last_selection_summary = self._run_lucb_feedback(client_models)
            else:
                feedback_seconds, self._last_selection_summary = self._run_legacy_full_scan_feedback(client_models)
        set_parameters(self.model, weights)
        self.time_log.append({
            "round": server_round, "round_time_sec": round(time.time() - self.round_start, 2),
            "aggregate_time_sec": round(aggregation_seconds, 2), "feedback_time_sec": round(feedback_seconds, 2),
            "feedback_triggered": triggered, **self._last_selection_summary,
        })
        return fl.common.ndarrays_to_parameters(weights), {}

    def evaluate(self, server_round: int, parameters: fl.common.Parameters) -> Tuple[float, Dict[str, float]]:
        """Record round zero only because aggregate_fit records later global models."""

        return self.evaluate_parameters(server_round, fl.common.parameters_to_ndarrays(parameters), log_result=(server_round == 0))

    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters,
                           client_manager: fl.server.client_manager.ClientManager) -> None:
        return None

    def save_reports(self, total_time: float) -> None:
        """Persist run-local summaries and configuration."""

        pd.DataFrame(self.metrics_log).to_csv(
            os.path.join(CONFIG["server_log_dir"], CONFIG["server_summary_filename"]), index=False
        )
        pd.DataFrame(self.time_log).to_csv(
            os.path.join(CONFIG["server_log_dir"], CONFIG["server_time_filename"]), index=False
        )
        config = dict(CONFIG)
        config["total_time_sec"] = round(total_time, 2)
        write_json(os.path.join(CONFIG["server_log_dir"], CONFIG["server_config_filename"]), config)
        print(f"[SERVER] Training summary saved to {CONFIG['server_log_dir']}")
        print(f"[SERVER] Global monitoring saved to {CONFIG['server_monitoring_path']}")
        print(f"[SERVER] Sanitized audit saved to {CONFIG['server_audit_dir']}")
        print(f"[SERVER] Total execution time: {total_time:.2f} seconds")


def log_system_info() -> None:
    info = {
        "Platform": platform.platform(), "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=False), "CPU Threads": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }
    print("[SERVER] System information:")
    for key, value in info.items():
        print(f"[SERVER] {key}: {value}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    apply_args(args)
    if CONFIG["feedback_protocol"] == "legacy":
        print("[SERVER] WARNING: Legacy feedback can transmit target-domain raw values; audit records remain sanitized.")
    started = time.time()
    os.makedirs(CONFIG["server_log_dir"], exist_ok=True)
    os.makedirs(CONFIG["server_feedback_dir"], exist_ok=True)
    os.makedirs(CONFIG["server_audit_dir"], exist_ok=True)
    print(f"[SERVER] Logging to {CONFIG['server_log_dir']}")
    print("[SERVER] Loading target-domain test data...")
    testloader, input_size, feature_names = load_test_data()
    log_system_info()
    model = SimpleNN(input_size)
    strategy = FeedbackFedAvg(
        model=model, testloader=testloader, feature_names=feature_names, input_size=input_size,
        min_fit_clients=CONFIG["min_clients"], min_available_clients=CONFIG["min_clients"],
        min_evaluate_clients=1, initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model)),
    )
    print("[SERVER] Starting Flower server...")
    fl.server.start_server(
        server_address=CONFIG["server_address"], config=fl.server.ServerConfig(num_rounds=CONFIG["num_rounds"]),
        strategy=strategy,
    )
    strategy.save_reports(time.time() - started)


if __name__ == "__main__":
    main()
