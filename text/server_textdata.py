"""Federated text server with budgeted LUCB feedback selection.

The server keeps target-domain text local. In the default abstract protocol it
only sends weak class identifiers and an aggregate distribution over coarse
salient positions. The legacy token protocol is deliberately opt-in.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("Value must be in the interval (0, 1].")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("Value must be in the interval (0, 1).")
    return parsed


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the text FL server with LUCB minibatch feedback selection."
    )
    parser.add_argument("--server-address", default="0.0.0.0:8080")
    parser.add_argument("--num-rounds", type=_positive_int, default=100)
    parser.add_argument("--ckpt-every", type=_positive_int, default=1)
    parser.add_argument("--min-clients", type=_positive_int, default=5)
    parser.add_argument("--test-csv", default="data/test_server.csv")
    parser.add_argument("--run-dir", default="runs/text_server")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Defaults to <run-dir>/checkpoints.",
    )
    parser.add_argument(
        "--audit-dir", default=None, help="Defaults to <run-dir>/audit."
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="JSON label-to-id mapping. The server creates one in the run directory by default.",
    )
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=_positive_int, default=64)
    parser.add_argument("--inference-batch-size", type=_positive_int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training/evaluation device: auto, cpu, cuda, or a valid torch device string.",
    )
    parser.add_argument(
        "--selection-device",
        default="cpu",
        help="Device used for client-model LUCB scoring and Grad-CAM attribution.",
    )
    parser.add_argument(
        "--max-cached-selection-models",
        type=_positive_int,
        default=1,
        help="Bound the number of client models retained while LUCB runs.",
    )

    parser.add_argument(
        "--selection-mode",
        choices=("lucb", "legacy_full_scan"),
        default="lucb",
    )
    parser.add_argument("--lucb-budget-fraction", type=_fraction, default=0.25)
    parser.add_argument("--lucb-batch-size", type=_positive_int, default=64)
    parser.add_argument(
        "--lucb-top-k",
        type=int,
        default=0,
        help="Number of weak clients to select; 0 chooses the shared automatic limit (at most 3).",
    )
    parser.add_argument("--lucb-epsilon", type=_probability, default=0.02)
    parser.add_argument("--lucb-delta", type=_probability, default=0.05)
    parser.add_argument("--weak-class-accuracy", type=_probability, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lucb-prioritization",
        choices=("strict", "decay_priority"),
        default="strict",
        help="Strict resets statistical evidence after client weights change.",
    )

    parser.add_argument(
        "--feedback-protocol", choices=("abstract", "legacy"), default="abstract"
    )
    parser.add_argument(
        "--allow-raw-feedback",
        action="store_true",
        help="Allow the legacy token protocol. This is rejected unless feedback-protocol=legacy.",
    )
    parser.add_argument(
        "--allow-partial-feedback",
        action="store_true",
        help="Deliver feedback to identified weak classes even if budget was exhausted before full separation.",
    )
    parser.add_argument(
        "--feedback-trigger",
        choices=("auto", "always", "never"),
        default="auto",
        help="Auto uses local overfit and global plateau monitoring.",
    )
    parser.add_argument("--overfit-threshold", type=_probability, default=0.95)
    parser.add_argument("--plateau-patience", type=_positive_int, default=3)
    parser.add_argument("--plateau-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--max-feedback-samples-per-class", type=_positive_int, default=4
    )
    return parser


def _validate_csv_header(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = {"text", "label"}.difference(header)
    if missing:
        raise ValueError(
            f"Target CSV must contain text and label columns; missing: {sorted(missing)}."
        )


def validate_startup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve paths and reject unsafe protocol settings before model loading."""
    args.test_csv = _resolve_path(args.test_csv)
    args.run_dir = _resolve_path(args.run_dir)
    args.checkpoint_dir = (
        _resolve_path(args.checkpoint_dir)
        if args.checkpoint_dir
        else args.run_dir / "checkpoints"
    )
    args.audit_dir = _resolve_path(args.audit_dir) if args.audit_dir else args.run_dir / "audit"
    args.label_map = _resolve_path(args.label_map) if args.label_map else args.run_dir / "label_map.json"
    args.lucb_top_k = None if args.lucb_top_k == 0 else args.lucb_top_k

    if not args.test_csv.is_file():
        raise ValueError(f"Target CSV does not exist: {args.test_csv}")
    _validate_csv_header(args.test_csv)
    if args.feedback_protocol == "legacy" and not args.allow_raw_feedback:
        raise ValueError(
            "The legacy feedback protocol requires --allow-raw-feedback."
        )
    if args.feedback_protocol == "legacy":
        warnings.warn(
            "Legacy token feedback is enabled. Runtime payloads may contain token strings; "
            "audit records remain sanitized.",
            RuntimeWarning,
        )
    if args.allow_raw_feedback and args.feedback_protocol != "legacy":
        raise ValueError(
            "--allow-raw-feedback is only valid with --feedback-protocol legacy."
        )
    if args.lucb_top_k is not None and args.lucb_top_k <= 0:
        raise ValueError("--lucb-top-k must be zero (automatic) or greater than zero.")
    if args.plateau_tolerance < 0.0:
        raise ValueError("--plateau-tolerance must not be negative.")
    return args


def _load_or_create_label_map(labels: Sequence[object], path: Path) -> dict[str, int]:
    labels_as_text = [str(label) for label in labels]
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            raw_mapping = json.load(handle)
        if not isinstance(raw_mapping, dict):
            raise ValueError(f"Label map must be a JSON object: {path}")
        mapping = {str(key): int(value) for key, value in raw_mapping.items()}
        missing = sorted(set(labels_as_text).difference(mapping))
        if missing:
            raise ValueError(
                f"Label map {path} does not define target labels: {missing[:10]}"
            )
        return mapping

    mapping = {label: index for index, label in enumerate(sorted(set(labels_as_text)))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[Server] Created stable label map at {path}")
    return mapping


def _label_map_fingerprint(mapping: Mapping[str, int]) -> str:
    """Return a stable identifier used to reject mismatched client label maps."""

    canonical = json.dumps(
        {str(key): int(value) for key, value in mapping.items()},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _select_device(torch_module: Any, raw_device: str) -> Any:
    if raw_device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    device = torch_module.device(raw_device)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise ValueError("A CUDA device was requested but CUDA is not available.")
    return device


def _fingerprint_models(client_parameters: Mapping[str, Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for client_id in sorted(client_parameters):
        digest.update(client_id.encode("utf-8"))
        for parameter in client_parameters[client_id]:
            digest.update(str(parameter.dtype).encode("ascii"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(parameter.tobytes())
    return digest.hexdigest()


def _stable_client_id(value: Any) -> str | None:
    """Accept only client identifiers safe to use in feedback audit records."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate if re.fullmatch(r"[A-Za-z0-9_.-]+", candidate) else None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "_asdict"):
        return {key: _json_safe(item) for key, item in value._asdict().items()}
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_path.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append metadata-only monitoring data without replacing prior rounds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


class _ClientModelCache:
    """Bounded cache that avoids retaining every post-fit text model in memory."""

    def __init__(
        self,
        max_models: int,
        make_model: Any,
        set_parameters: Any,
        parameters_by_client: Mapping[str, Sequence[Any]],
    ) -> None:
        self.max_models = max_models
        self.make_model = make_model
        self.set_parameters = set_parameters
        self.parameters_by_client = parameters_by_client
        self.models: OrderedDict[str, Any] = OrderedDict()

    def get(self, client_id: str) -> Any:
        if client_id in self.models:
            self.models.move_to_end(client_id)
            return self.models[client_id]
        model = self.make_model()
        self.set_parameters(model, self.parameters_by_client[client_id])
        self.models[client_id] = model
        while len(self.models) > self.max_models:
            _, evicted = self.models.popitem(last=False)
            del evicted
        return model

    def clear(self) -> None:
        self.models.clear()


def run_server(args: argparse.Namespace) -> None:
    """Load runtime dependencies only after deterministic startup validation."""
    try:
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import flwr as fl
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        from fl_lucb import (
            BatchScore,
            LUCBSelector,
            MinibatchPlan,
            SelectionConfig,
            SelectionState,
            build_audit_record,
            sanitize_audit_record,
            validate_feedback_payload,
        )
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Could not load the text runtime dependencies. Install the pinned text/requirements.txt "
            "in a compatible environment and ensure the repository root is importable."
        ) from exc

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.test_csv)
    frame = frame.dropna(subset=["text", "label"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Target CSV has no usable rows after removing missing text or labels.")
    label_map = _load_or_create_label_map(frame["label"].tolist(), args.label_map)
    frame["label_id"] = frame["label"].map(lambda item: label_map[str(item)])
    num_labels = max(label_map.values()) + 1
    label_map_fingerprint = _label_map_fingerprint(label_map)
    if num_labels < 2:
        raise ValueError("At least two labels are required for sequence classification.")

    device = _select_device(torch, args.device)
    selection_device = _select_device(torch, args.selection_device)
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    encoded = tokenizer(
        frame["text"].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt",
    )
    target_labels = frame["label_id"].astype(int).tolist()

    class DistilBertClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = DistilBertForSequenceClassification.from_pretrained(
                args.model_name, num_labels=num_labels
            )

        def forward(self, input_ids: Any, attention_mask: Any = None, labels: Any = None) -> Any:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

    def make_model(on_device: Any) -> Any:
        return DistilBertClassifier().to(on_device)

    def get_parameters(model: Any) -> list[Any]:
        return [item.detach().cpu().numpy() for item in model.state_dict().values()]

    def set_parameters(model: Any, parameters: Sequence[Any]) -> None:
        state_dict = model.state_dict()
        if len(state_dict) != len(parameters):
            raise ValueError(
                f"Parameter length mismatch: model expects {len(state_dict)}, received {len(parameters)}."
            )
        for (key, existing), parameter in zip(state_dict.items(), parameters):
            state_dict[key] = torch.tensor(parameter, dtype=existing.dtype)
        model.load_state_dict(state_dict, strict=True)

    def predict_sample_ids(model: Any, sample_ids: Sequence[int]) -> list[int]:
        if not sample_ids:
            return []
        model.eval()
        model_device = next(model.parameters()).device
        predictions: list[int] = []
        with torch.no_grad():
            for start in range(0, len(sample_ids), args.inference_batch_size):
                batch_ids = [int(item) for item in sample_ids[start : start + args.inference_batch_size]]
                index_tensor = torch.tensor(batch_ids, dtype=torch.long)
                input_ids = encoded["input_ids"].index_select(0, index_tensor).to(model_device)
                attention_mask = encoded["attention_mask"].index_select(0, index_tensor).to(model_device)
                logits = model(input_ids, attention_mask=attention_mask).logits
                predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        return [int(item) for item in predictions]

    def evaluate_global(parameters: Sequence[Any], server_round: int) -> float:
        set_parameters(global_model, parameters)
        predictions = predict_sample_ids(global_model, list(range(len(frame))))
        correct = sum(prediction == label for prediction, label in zip(predictions, target_labels))
        accuracy = correct / len(target_labels)
        print(f"[Server] Round {server_round}: global target accuracy={accuracy:.4f}")
        return accuracy

    def attribution_for_sample(model: Any, sample_id: int, include_tokens: bool) -> list[dict[str, Any]]:
        """Return coarse positions unless legacy token feedback was explicitly enabled."""
        model.eval()
        model_device = next(model.parameters()).device
        index_tensor = torch.tensor([int(sample_id)], dtype=torch.long)
        input_ids = encoded["input_ids"].index_select(0, index_tensor).to(model_device)
        attention_mask = encoded["attention_mask"].index_select(0, index_tensor).to(model_device)
        embedding_output: Any = None

        def capture_embeddings(_module: Any, _inputs: Any, output: Any) -> None:
            nonlocal embedding_output
            embedding_output = output

        hook = model.model.distilbert.embeddings.register_forward_hook(capture_embeddings)
        try:
            model.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask=attention_mask).logits
            predicted_label = int(logits.argmax(dim=1).item())
            if embedding_output is None:
                return []
            embedding_output.retain_grad()
            logits[0, predicted_label].backward()
            gradients = embedding_output.grad.squeeze(0).norm(dim=1).detach().cpu().tolist()
            token_ids = input_ids.squeeze(0).detach().cpu().tolist()
            ignored_tokens = {"[CLS]", "[SEP]", "[PAD]"}
            special_ids = set(tokenizer.all_special_ids)
            scored: list[tuple[int, float, str | None]] = []
            for index, (token_id, score) in enumerate(zip(token_ids, gradients)):
                token = tokenizer.convert_ids_to_tokens(int(token_id)) if include_tokens else None
                if include_tokens and token in ignored_tokens:
                    continue
                if not include_tokens and int(token_id) in special_ids:
                    continue
                scored.append((index, float(score), token))
            if not scored:
                return []
            maximum = max(score for _, score, _ in scored)
            threshold = max(0.35 * maximum, 1e-9)
            selected = [item for item in scored if item[1] >= threshold]
            selected.sort(key=lambda item: item[1], reverse=True)
            if len(selected) < 3:
                for item in scored:
                    if item not in selected:
                        selected.append(item)
                    if len(selected) >= 3:
                        break
            selected = selected[:10]
            total_tokens = max(1, len(token_ids))
            output: list[dict[str, Any]] = []
            for index, score, token in selected:
                if index / total_tokens < 1 / 3:
                    position = "beginning"
                elif index / total_tokens < 2 / 3:
                    position = "middle"
                else:
                    position = "end"
                item: dict[str, Any] = {"position": position, "weight": score}
                if include_tokens and token is not None:
                    item["token"] = token
                output.append(item)
            return output
        finally:
            hook.remove()

    def class_assessments_to_audit(assessments: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        audit: dict[str, list[dict[str, Any]]] = {}
        for client_id, rows in assessments.items():
            audit[str(client_id)] = []
            for row in rows:
                audit[str(client_id)].append(
                    {
                        "label": int(row.label),
                        "accuracy": float(row.accuracy),
                        "lower": float(row.lower),
                        "upper": float(row.upper),
                        "status": str(row.status),
                    }
                )
        return audit

    global_model = make_model(device)

    def checkpoint_path(round_index: int) -> Path:
        return args.checkpoint_dir / f"round_{round_index}.pt"

    def load_latest_checkpoint() -> dict[str, Any] | None:
        checkpoint_files = sorted(
            args.checkpoint_dir.glob("round_*.pt"),
            key=lambda item: int(item.stem.split("_")[1]),
        )
        if not checkpoint_files:
            return None
        latest = checkpoint_files[-1]
        try:
            checkpoint = torch.load(latest, map_location="cpu")
        except (pickle.UnpicklingError, RuntimeError):
            warnings.warn(
                f"Loading legacy checkpoint with pickle compatibility enabled: {latest}",
                RuntimeWarning,
            )
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        print(f"[Server] Loaded checkpoint {latest}")
        return checkpoint

    def save_checkpoint(parameters: Any, round_index: int, history: list[dict[str, Any]]) -> None:
        path = checkpoint_path(round_index)
        torch.save({"round": round_index, "parameters": parameters, "history": history}, path)
        print(f"[Server] Saved checkpoint {path}")

    checkpoint = load_latest_checkpoint()
    if checkpoint:
        checkpoint_parameters = checkpoint["parameters"]
        parameters_as_arrays = (
            parameters_to_ndarrays(checkpoint_parameters)
            if hasattr(checkpoint_parameters, "tensors")
            else checkpoint_parameters
        )
        set_parameters(global_model, parameters_as_arrays)
        initial_parameters = ndarrays_to_parameters(parameters_as_arrays)
        start_round = int(checkpoint["round"]) + 1
        prior_history = list(checkpoint.get("history", []))
    else:
        initial_parameters = ndarrays_to_parameters(get_parameters(global_model))
        start_round = 1
        prior_history: list[dict[str, Any]] = []

    selection_config = SelectionConfig(
        selection_mode="lucb",
        lucb_budget_fraction=args.lucb_budget_fraction,
        lucb_batch_size=args.lucb_batch_size,
        lucb_top_k=args.lucb_top_k,
        lucb_epsilon=args.lucb_epsilon,
        lucb_delta=args.lucb_delta,
        weak_class_accuracy=args.weak_class_accuracy,
        seed=args.seed,
        lucb_prioritization=args.lucb_prioritization,
        allow_partial_feedback=bool(args.allow_partial_feedback),
    )

    class LUCBTextFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.history = prior_history
            self.global_round_done = start_round - 1
            self.pending_feedback: dict[str, dict[str, Any]] = {}
            self._selection_state = SelectionState()
            self._current_parameters = initial_parameters

        def _feedback_reason(self) -> str | None:
            if args.feedback_trigger == "never":
                return None
            if args.feedback_trigger == "always":
                return "forced"
            if not self.history:
                return None
            latest_train_accuracy = float(
                self.history[-1].get(
                    "mean_client_train_accuracy", self.history[-1].get("train", 0.0)
                )
            )
            if latest_train_accuracy > args.overfit_threshold:
                return "overfit"
            needed = args.plateau_patience + 1
            if len(self.history) >= needed:
                recent = [
                    entry.get("global_target_accuracy", entry.get("test"))
                    for entry in self.history[-needed:]
                ]
                if all(value is not None for value in recent) and (
                    max(recent) - min(recent) <= args.plateau_tolerance
                ):
                    return "global_plateau"
            return None

        def _legacy_full_scan(
            self,
            client_parameters: Mapping[str, Sequence[Any]],
            model_fingerprint: str,
            cache: _ClientModelCache,
        ) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, dict[int, dict[str, int]]], dict[str, Any]]:
            plans = {
                client_id: MinibatchPlan.build(
                    client_id=client_id,
                    labels=target_labels,
                    batch_size=args.lucb_batch_size,
                    seed=args.seed,
                    sample_ids=list(range(len(frame))),
                )
                for client_id in sorted(client_parameters)
            }
            observed: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)

            def score_batch(client_id: str, batch: Any) -> Any:
                sample_ids = [int(item) for item in batch.sample_ids]
                predictions = predict_sample_ids(cache.get(client_id), sample_ids)
                labels = [int(item) for item in batch.labels]
                for sample_id, label, prediction in zip(sample_ids, labels, predictions):
                    observed[client_id][sample_id] = {"label": label, "prediction": prediction}
                return BatchScore.from_predictions(labels, predictions)

            legacy_config = SelectionConfig(
                selection_mode="legacy_full_scan",
                lucb_budget_fraction=1.0,
                lucb_batch_size=args.lucb_batch_size,
                lucb_top_k=args.lucb_top_k,
                lucb_epsilon=args.lucb_epsilon,
                lucb_delta=args.lucb_delta,
                weak_class_accuracy=args.weak_class_accuracy,
                seed=args.seed,
                lucb_prioritization="strict",
            )
            full_scan_budget = sum(plan.total_samples for plan in plans.values())
            result = LUCBSelector(legacy_config).select(
                plans,
                score_batch,
                model_fingerprint=model_fingerprint,
                state=None,
                budget=full_scan_budget,
            )
            return self._selection_result_parts(result, observed)

        def _selection_result_parts(
            self,
            result: Any,
            observed: Mapping[str, Mapping[int, Mapping[str, int]]],
        ) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, dict[int, dict[str, int]]], dict[str, Any]]:
            class_summaries = class_assessments_to_audit(result.selected_classes)
            summary = {
                "status": str(result.status),
                "stop_reason": str(result.stop_reason),
                "feedback_ready": bool(result.feedback_ready),
                "model_sample_evaluations": int(result.model_sample_evaluations),
                "budget": int(result.budget),
                "confidence_bounds": result.confidence_bounds,
                "coverage": result.coverage,
                "scored_batches": result.scored_batches,
                "boundary_gap": result.boundary_gap,
            }
            return list(result.selected_client_ids), class_summaries, dict(observed), summary

        def _run_lucb_selection(
            self,
            client_parameters: Mapping[str, Sequence[Any]],
            model_fingerprint: str,
            cache: _ClientModelCache,
        ) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, dict[int, dict[str, int]]], dict[str, Any]]:
            plans = {
                client_id: MinibatchPlan.build(
                    client_id=client_id,
                    labels=target_labels,
                    batch_size=args.lucb_batch_size,
                    seed=args.seed,
                    sample_ids=list(range(len(frame))),
                )
                for client_id in sorted(client_parameters)
            }
            observed: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)

            def score_batch(client_id: str, batch: Any) -> Any:
                sample_ids = [int(item) for item in batch.sample_ids]
                predictions = predict_sample_ids(cache.get(client_id), sample_ids)
                labels = [int(item) for item in batch.labels]
                for sample_id, label, prediction in zip(sample_ids, labels, predictions):
                    observed[client_id][sample_id] = {"label": label, "prediction": prediction}
                return BatchScore.from_predictions(labels, predictions)

            state = None if args.lucb_prioritization == "strict" else self._selection_state
            result = LUCBSelector(selection_config).select(
                plans,
                score_batch,
                model_fingerprint=model_fingerprint,
                state=state,
            )
            if args.lucb_prioritization == "strict":
                self._selection_state = SelectionState()
            else:
                self._selection_state = state
            return self._selection_result_parts(result, observed)

        def _build_payloads(
            self,
            selected_clients: Sequence[str],
            class_summaries: Mapping[str, Sequence[Mapping[str, Any]]],
            observed: Mapping[str, Mapping[int, Mapping[str, int]]],
            cache: _ClientModelCache,
        ) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, list[int]]]:
            payloads: dict[str, dict[str, Any]] = {}
            explanation_counts: dict[str, int] = {}
            weak_labels_by_client: dict[str, list[int]] = {}
            include_tokens = args.feedback_protocol == "legacy"
            for client_id in selected_clients:
                weak_labels = sorted(
                    int(row["label"])
                    for row in class_summaries.get(client_id, [])
                    if row.get("status") == "weak"
                )
                weak_labels_by_client[client_id] = weak_labels
                if not weak_labels:
                    continue
                candidates_by_label: dict[int, list[int]] = defaultdict(list)
                for sample_id, outcome in observed.get(client_id, {}).items():
                    label = int(outcome["label"])
                    if label in weak_labels and int(outcome["prediction"]) != label:
                        candidates_by_label[label].append(int(sample_id))
                position_totals: dict[str, float] = defaultdict(float)
                legacy_tokens: list[dict[str, Any]] = []
                explanation_count = 0
                model = cache.get(client_id)
                for label in weak_labels:
                    for sample_id in sorted(candidates_by_label[label])[: args.max_feedback_samples_per_class]:
                        explanation = attribution_for_sample(model, sample_id, include_tokens=include_tokens)
                        if not explanation:
                            continue
                        explanation_count += 1
                        for item in explanation:
                            position_totals[str(item["position"])] += float(item["weight"])
                        if include_tokens:
                            tokens = [
                                str(item["token"])
                                for item in explanation
                                if item.get("token")
                            ]
                            if tokens:
                                legacy_tokens.append({"label": label, "tokens": tokens})
                if explanation_count == 0:
                    continue
                if include_tokens:
                    payload: dict[str, Any] = {
                        "protocol": "legacy",
                        "domain": "text",
                        "weak_labels": weak_labels,
                        "token_feedback": legacy_tokens,
                    }
                else:
                    total_weight = sum(position_totals.values())
                    salient_positions = [
                        {"position": position, "weight": weight / total_weight}
                        for position, weight in sorted(position_totals.items())
                        if total_weight > 0.0
                    ]
                    if not salient_positions:
                        continue
                    payload = {
                        "protocol": "abstract",
                        "domain": "text",
                        "weak_labels": weak_labels,
                        "salient_positions": salient_positions,
                    }
                payloads[client_id] = validate_feedback_payload(
                    payload,
                    protocol=args.feedback_protocol,
                    allow_raw_feedback=args.allow_raw_feedback,
                    domain="text",
                )
                explanation_counts[client_id] = explanation_count
            return payloads, explanation_counts, weak_labels_by_client

        def _write_feedback_audit(
            self,
            round_index: int,
            reason: str,
            fingerprint: str,
            selection: Mapping[str, Any],
            selected_clients: Sequence[str],
            class_summaries: Mapping[str, Sequence[Mapping[str, Any]]],
            payloads: Mapping[str, Mapping[str, Any]],
            explanation_counts: Mapping[str, int],
            weak_labels_by_client: Mapping[str, Sequence[int]],
            monitoring: Mapping[str, Any],
        ) -> None:
            record = build_audit_record(
                event_id=f"text-feedback-round-{round_index}",
                configuration={
                        "selection_mode": args.selection_mode,
                        "lucb_budget_fraction": (
                            1.0 if args.selection_mode == "legacy_full_scan" else args.lucb_budget_fraction
                        ),
                        "lucb_batch_size": args.lucb_batch_size,
                        "lucb_top_k": args.lucb_top_k,
                        "lucb_epsilon": args.lucb_epsilon,
                        "lucb_delta": args.lucb_delta,
                        "weak_class_accuracy": args.weak_class_accuracy,
                        "seed": args.seed,
                        "lucb_prioritization": args.lucb_prioritization,
                        "feedback_protocol": args.feedback_protocol,
                },
                model_fingerprint=fingerprint,
                status=str(selection["status"]),
                stop_reason=str(selection["stop_reason"]),
                model_sample_evaluations=int(selection["model_sample_evaluations"]),
                budget=int(selection["budget"]),
                selected_client_ids=list(selected_clients),
                selected_classes=class_summaries,
                confidence_bounds=selection.get("confidence_bounds", {}),
                coverage=selection.get("coverage", {}),
                scored_batches=selection.get("scored_batches", {}),
                boundary_gap=selection.get("boundary_gap"),
                explanation_cost={
                    "attribution_model_sample_evaluations": sum(explanation_counts.values()),
                    "by_client": dict(explanation_counts),
                },
                protocol=args.feedback_protocol,
                metadata={
                    "round": round_index,
                    "trigger_reason": reason,
                    "weak_labels": {key: list(value) for key, value in weak_labels_by_client.items()},
                    "global_monitoring": dict(monitoring),
                    "feedback_delivered": sorted(payloads),
                    "feedback_outcome": selection.get("feedback_outcome", "not_attempted"),
                    "feedback_outcome_reason": selection.get("feedback_outcome_reason"),
                },
            )
            safe_record = sanitize_audit_record(record, protocol=args.feedback_protocol)
            _write_json(
                args.audit_dir / f"lucb_audit_feedback_round_{round_index}.json",
                safe_record,
            )

        def aggregate_fit(self, server_round: int, results: Any, failures: Any) -> Any:
            self.global_round_done += 1
            global_round = self.global_round_done
            if not results:
                warnings.warn("No client returned fit results; skipping the round.")
                _append_jsonl(
                    args.run_dir / "feedback_status.jsonl",
                    {
                        "round": global_round,
                        "status": "skipped",
                        "stop_reason": "no_client_results",
                    },
                )
                return self._current_parameters, {}
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is None:
                warnings.warn("FedAvg returned no aggregated parameters; skipping the round.")
                return self._current_parameters, {}
            aggregated_parameters, aggregated_metrics = aggregated
            self._current_parameters = aggregated_parameters
            arrays = parameters_to_ndarrays(aggregated_parameters)
            global_accuracy = evaluate_global(arrays, global_round)
            train_accuracies = [
                float(fit_result.metrics.get("train_accuracy", 0.0))
                for _proxy, fit_result in results
            ]
            monitoring = {
                "global_target_accuracy": global_accuracy,
                "mean_client_train_accuracy": float(np.mean(train_accuracies)),
                "global_monitoring_model_sample_evaluations": len(frame),
                "included_in_lucb_budget": False,
            }
            self.history.append(monitoring)
            _append_jsonl(
                args.run_dir / "global_monitoring.jsonl",
                {"round": global_round, **monitoring},
            )

            reason = self._feedback_reason()
            if reason:
                client_parameters: dict[str, Sequence[Any]] = {}
                proxy_id_by_client: dict[str, str] = {}
                duplicate_client_ids: list[str] = []
                for proxy, fit_result in results:
                    reported_id = _stable_client_id(fit_result.metrics.get("client_id"))
                    stable_id = reported_id or f"proxy-{proxy.cid}"
                    if reported_id is None:
                        warnings.warn(
                            f"Client proxy {proxy.cid} did not report a valid stable client_id; "
                            "using a session-scoped fallback for this feedback event.",
                            RuntimeWarning,
                        )
                    if stable_id in client_parameters:
                        duplicate_client_ids.append(stable_id)
                        continue
                    client_parameters[stable_id] = parameters_to_ndarrays(fit_result.parameters)
                    proxy_id_by_client[stable_id] = str(proxy.cid)
                if duplicate_client_ids:
                    self.pending_feedback = {}
                    _append_jsonl(
                        args.run_dir / "feedback_status.jsonl",
                        {
                            "round": global_round,
                            "status": "skipped",
                            "stop_reason": "duplicate_stable_client_id",
                            "duplicate_client_ids": sorted(set(duplicate_client_ids)),
                        },
                    )
                    warnings.warn(
                        "Skipping feedback because multiple fit results reported the same stable client_id: "
                        f"{sorted(set(duplicate_client_ids))}.",
                        RuntimeWarning,
                    )
                else:
                    fingerprint = _fingerprint_models(client_parameters)
                    cache = _ClientModelCache(
                        args.max_cached_selection_models,
                        lambda: make_model(selection_device),
                        set_parameters,
                        client_parameters,
                    )
                    try:
                        if args.selection_mode == "lucb":
                            selected, summaries, observed, selection = self._run_lucb_selection(
                                client_parameters, fingerprint, cache
                            )
                        else:
                            selected, summaries, observed, selection = self._legacy_full_scan(
                                client_parameters, fingerprint, cache
                            )
                        if selection["feedback_ready"]:
                            payloads, explanation_counts, weak_labels = self._build_payloads(
                                selected, summaries, observed, cache
                            )
                            selection["feedback_outcome"] = (
                                "delivered" if payloads else "not_delivered"
                            )
                            selection["feedback_outcome_reason"] = (
                                "payloads_ready"
                                if payloads
                                else "no_confident_weak_class_or_no_observed_error"
                            )
                            self.pending_feedback = {
                                proxy_id_by_client[client_id]: payload
                                for client_id, payload in payloads.items()
                            }
                        else:
                            payloads = {}
                            explanation_counts = {}
                            weak_labels = {}
                            selection["feedback_outcome"] = "not_delivered"
                            selection["feedback_outcome_reason"] = (
                                "selection_not_complete_or_no_confident_weak_class"
                            )
                            self.pending_feedback = {}
                        self._write_feedback_audit(
                            global_round,
                            reason,
                            fingerprint,
                            selection,
                            selected,
                            summaries,
                            payloads,
                            explanation_counts,
                            weak_labels,
                            monitoring,
                        )
                        transport_record = {
                            "round": global_round,
                            "protocol": args.feedback_protocol,
                            "recipients": payloads,
                        }
                        _write_json(args.run_dir / "feedback" / f"round_{global_round}.json", transport_record)
                        print(
                            f"[Server] Feedback event at round {global_round}: "
                            f"reason={reason}, selected={selected}, delivered={sorted(payloads)}"
                        )
                    finally:
                        cache.clear()

            if global_round % args.ckpt_every == 0:
                save_checkpoint(aggregated_parameters, global_round, self.history)
            return aggregated_parameters, aggregated_metrics

        def configure_fit(self, server_round: int, parameters: Any, client_manager: Any) -> Any:
            instructions = super().configure_fit(server_round, parameters, client_manager)
            for client_proxy, fit_instructions in instructions:
                fit_instructions.config["server_round"] = server_round
                fit_instructions.config["global_round"] = self.global_round_done + 1
                fit_instructions.config["feedback_protocol"] = args.feedback_protocol
                fit_instructions.config["label_map_fingerprint"] = label_map_fingerprint
                fit_instructions.config["num_labels"] = num_labels
                payload = self.pending_feedback.get(str(client_proxy.cid))
                if payload is not None:
                    fit_instructions.config["feedback_payload"] = json.dumps(payload, ensure_ascii=False)
            self.pending_feedback = {}
            return instructions

    strategy = LUCBTextFedAvg(
        evaluate_fn=None,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
    )
    remaining_rounds = args.num_rounds - (start_round - 1)
    if remaining_rounds <= 0:
        print(
            f"[Server] Checkpoint already reached requested total rounds={args.num_rounds}; nothing to run."
        )
        return
    print(
        f"[Server] Starting text FL at {args.server_address}; "
        f"total_rounds={args.num_rounds}, additional_rounds={remaining_rounds}, "
        f"selection={args.selection_mode}, protocol={args.feedback_protocol}."
    )
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=remaining_rounds),
        strategy=strategy,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    try:
        args = validate_startup_args(parser.parse_args(argv))
        run_server(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
