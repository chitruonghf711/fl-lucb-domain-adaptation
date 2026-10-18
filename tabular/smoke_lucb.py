"""Fast deterministic LUCB smoke test for the tabular feedback contract.

This helper uses five synthetic client scorers and no Flower server, model
training, dataset, TVAE, or SHAP run. It verifies the budget, deterministic
selection, abstract payload schema, and sanitized audit output on Windows or
any standard Python environment.

    python tabular/smoke_lucb.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl_lucb import (
    BatchScore,
    LUCBSelector,
    MinibatchPlan,
    SelectionConfig,
    validate_feedback_payload,
    write_audit_json,
)


def _contains_sensitive_key(value: Any) -> bool:
    """Detect fields that must never occur in the abstract audit record."""

    forbidden = ("raw", "reference", "value", "row", "token", "text", "pixel", "tensor", "input")
    if isinstance(value, dict):
        return any(
            any(part in str(key).lower() for part in forbidden) or _contains_sensitive_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sensitive_key(child) for child in value)
    return False


def _plans(seed: int) -> Dict[str, MinibatchPlan]:
    """Create five deterministic, class-balanced synthetic target schedules."""

    labels = [index % 2 for index in range(256)]
    return {
        f"client_{index}": MinibatchPlan.build(
            client_id=f"client_{index}", labels=labels, sample_ids=list(range(len(labels))),
            batch_size=16, seed=seed,
        )
        for index in range(1, 6)
    }


def _scorer(error_rates: Dict[str, float]):
    """Return deterministic predictions based solely on the requested batch IDs."""

    def score(client_id: str, batch: Any) -> BatchScore:
        predictions = []
        for sample_id, label in zip(batch.sample_ids, batch.labels):
            cutoff = int(error_rates[client_id] * 100)
            wrong = int(sample_id) % 100 < cutoff
            predictions.append(1 - int(label) if wrong else int(label))
        return BatchScore.from_predictions(batch.labels, predictions)

    return score


def _abstract_payload(client_id: str, labels: Iterable[int]) -> Dict[str, Any]:
    """Build a representative row-free tabular payload for schema verification."""

    return validate_feedback_payload(
        {
            "protocol": "abstract",
            "domain": "tabular",
            "client_id": client_id,
            "weak_labels": sorted(set(labels)),
            "feature_ranking": [
                {"feature": "0", "rank": 1, "importance": 0.42},
                {"feature": "3", "rank": 2, "importance": 0.18},
            ],
        },
        protocol="abstract",
        domain="tabular",
    )


def run_smoke(seed: int = 42) -> Dict[str, Any]:
    """Run the five-client selection, payload, and audit checks."""

    plans = _plans(seed)
    selector = LUCBSelector(SelectionConfig(
        selection_mode="lucb",
        lucb_budget_fraction=0.25,
        lucb_batch_size=16,
        lucb_top_k=3,
        lucb_epsilon=0.02,
        lucb_delta=0.05,
        weak_class_accuracy=0.85,
        seed=seed,
        lucb_prioritization="strict",
    ))
    result = selector.select(
        plans,
        _scorer({"client_1": 0.08, "client_2": 0.22, "client_3": 0.52, "client_4": 0.73, "client_5": 0.88}),
        model_fingerprint="smoke-model-v1",
    )
    if result.model_sample_evaluations > result.budget:
        raise AssertionError("LUCB model-sample evaluations exceeded the configured budget")
    if not result.selected_client_ids:
        raise AssertionError("LUCB did not select any weak synthetic client")
    selected_labels = [
        assessment.label
        for assessments in result.selected_classes.values()
        for assessment in assessments
    ]
    payload = _abstract_payload(result.selected_client_ids[0], selected_labels or [0])
    with tempfile.TemporaryDirectory(prefix="tabular_lucb_smoke_") as directory:
        audit_path = write_audit_json(Path(directory) / "audit.json", result.audit)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if _contains_sensitive_key(audit):
        raise AssertionError("Sanitized audit contains a prohibited raw-content field")
    return {
        "status": result.status,
        "stop_reason": result.stop_reason,
        "selected_client_ids": list(result.selected_client_ids),
        "model_sample_evaluations": result.model_sample_evaluations,
        "budget": result.budget,
        "weak_labels": payload["weak_labels"],
        "audit_sanitized": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), sort_keys=True))
