"""Audit construction and privacy sanitization for LUCB feedback events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .models import ClassAssessment, ScoredBatch, WilsonInterval, json_safe


class _DropValue:
    pass


DROP = _DropValue()

# These fields can carry source rows, token sequences, or image tensors. Their
# absence from abstract audit output is enforced recursively, including inside
# arbitrary integration metadata.
_SENSITIVE_FIELD_PARTS = (
    "raw",
    "reference",
    "feature_value",
    "row_data",
    "token",
    "text",
    "pixel",
    "image_data",
    "tensor",
    "input_data",
    "sample_data",
    "payload",
)
_SENSITIVE_EXACT_FIELDS = {
    "rows",
    "image",
    "images",
    "values",
    "samples",
    "inputs",
    "features",
}


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    if normalized in _SENSITIVE_EXACT_FIELDS:
        return True
    return any(part in normalized for part in _SENSITIVE_FIELD_PARTS)


def _sanitize(value: Any, protocol: str, key: Any | None = None) -> Any:
    # Audit storage is always metadata-only, including when runtime feedback
    # intentionally uses the legacy protocol.
    if key is not None and _is_sensitive_key(key):
        return DROP
    if isinstance(value, Mapping):
        cleaned = {}
        for child_key, child_value in value.items():
            safe_value = _sanitize(child_value, protocol, child_key)
            if safe_value is not DROP:
                cleaned[str(child_key)] = safe_value
        return cleaned
    if isinstance(value, (list, tuple, set)):
        cleaned_items = []
        for item in value:
            safe_item = _sanitize(item, protocol)
            if safe_item is not DROP:
                cleaned_items.append(safe_item)
        return cleaned_items
    return json_safe(value)


def sanitize_audit_record(
    record: Mapping[str, Any], protocol: str = "abstract"
) -> dict[str, Any]:
    """Return a JSON-safe audit record without raw content in abstract mode.

    Audit records remain metadata-only for both protocols. Legacy may transport
    raw feedback at runtime after explicit validation, but raw content is never
    copied into this record.
    """

    if protocol not in {"abstract", "legacy"}:
        raise ValueError("protocol must be 'abstract' or 'legacy'")
    if not isinstance(record, Mapping):
        raise ValueError("audit record must be a mapping")
    sanitized = _sanitize(record, protocol)
    if not isinstance(sanitized, dict):
        raise ValueError("audit record must serialize to an object")
    return sanitized


def _assessment_dict(assessment: ClassAssessment | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(assessment, ClassAssessment):
        return assessment.to_dict()
    if isinstance(assessment, Mapping):
        return sanitize_audit_record(assessment, protocol="abstract")
    raise ValueError("selected_classes must contain ClassAssessment or mapping values")


def _interval_dict(interval: WilsonInterval | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(interval, WilsonInterval):
        return interval.to_dict()
    if isinstance(interval, Mapping):
        return sanitize_audit_record(interval, protocol="abstract")
    raise ValueError("confidence_bounds must contain WilsonInterval or mapping values")


def _batch_dict(batch: ScoredBatch | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(batch, ScoredBatch):
        return batch.to_audit_dict()
    if isinstance(batch, Mapping):
        return sanitize_audit_record(batch, protocol="abstract")
    raise ValueError("scored_batches must contain ScoredBatch or mapping values")


def build_audit_record(
    record: Mapping[str, Any] | None = None,
    *,
    event_id: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    model_fingerprint: str | None = None,
    status: str | None = None,
    stop_reason: str | None = None,
    model_sample_evaluations: int | None = None,
    budget: int | None = None,
    selected_client_ids: Sequence[str] | None = None,
    selected_classes: Mapping[str, Sequence[ClassAssessment | Mapping[str, Any]]] | None = None,
    confidence_bounds: Mapping[str, WilsonInterval | Mapping[str, Any]] | None = None,
    coverage: Mapping[str, Mapping[str, Any]] | None = None,
    scored_batches: Mapping[str, Sequence[ScoredBatch | Mapping[str, Any]]] | None = None,
    boundary_gap: float | None = None,
    explanation_cost: Mapping[str, Any] | int | float | None = None,
    protocol: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    domain: str | None = None,
    feedback_protocol: str | None = None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one serializable selection audit event.

    The record includes only aggregate evidence and sampled identifiers. In
    abstract mode, sanitization removes any accidental raw feedback fields.
    """

    if record is not None:
        if selection is not None:
            raise ValueError("record and selection cannot be supplied together")
        if any(
            value is not None
            for value in (
                configuration,
                model_fingerprint,
                status,
                stop_reason,
                model_sample_evaluations,
                budget,
                selected_client_ids,
                selected_classes,
                confidence_bounds,
                coverage,
                scored_batches,
            )
        ):
            raise ValueError("record cannot be mixed with structured audit fields")
        return sanitize_audit_record(record, protocol=protocol or feedback_protocol or "abstract")

    if selection is not None:
        if not isinstance(selection, Mapping):
            raise ValueError("selection must be a mapping")
        event_id = event_id if event_id is not None else selection.get("event_id")
        model_fingerprint = model_fingerprint or str(selection.get("model_fingerprint", ""))
        status = status or str(selection.get("status", "unknown"))
        stop_reason = stop_reason or str(selection.get("stop_reason", "unknown"))
        model_sample_evaluations = (
            model_sample_evaluations
            if model_sample_evaluations is not None
            else int(selection.get("model_sample_evaluations", 0))
        )
        budget = budget if budget is not None else int(selection.get("budget", 0))
        selected_client_ids = selected_client_ids or selection.get("selected_client_ids", ())
        selected_classes = selected_classes or selection.get("selected_classes", {})
        confidence_bounds = confidence_bounds or selection.get("confidence_bounds", {})
        coverage = coverage or selection.get("coverage", {})
        scored_batches = scored_batches or selection.get("scored_batches", {})
        boundary_gap = boundary_gap if boundary_gap is not None else selection.get("boundary_gap")

    if model_sample_evaluations is None or budget is None:
        raise ValueError("model_sample_evaluations and budget are required")
    if model_sample_evaluations < 0 or budget < 0:
        raise ValueError("model_sample_evaluations and budget must be non-negative")
    if model_sample_evaluations > budget:
        raise ValueError("model_sample_evaluations cannot exceed budget")

    record: dict[str, Any] = {
        "schema_version": 1,
        "event_type": "lucb_selection",
        "event_id": event_id,
        "protocol": protocol or feedback_protocol or "abstract",
        "configuration": dict(configuration or {}),
        "model_fingerprint": model_fingerprint or "",
        "status": status or "unknown",
        "stop_reason": stop_reason or "unknown",
        "model_sample_evaluations": model_sample_evaluations,
        "budget": budget,
        "explanation_cost": explanation_cost if explanation_cost is not None else {},
        "selected_clients": list(selected_client_ids or ()),
        "selected_classes": {
            client_id: [_assessment_dict(item) for item in assessments]
            for client_id, assessments in (selected_classes or {}).items()
        },
        "confidence_bounds": {
            client_id: _interval_dict(interval)
            for client_id, interval in (confidence_bounds or {}).items()
        },
        "coverage": dict(coverage or {}),
        "scored_batches": {
            client_id: [_batch_dict(batch) for batch in batches]
            for client_id, batches in (scored_batches or {}).items()
        },
        "boundary_gap": boundary_gap,
        "metadata": dict(metadata or {}),
    }
    if domain is not None:
        record["domain"] = domain
    if feedback_protocol is not None:
        record["feedback_protocol"] = feedback_protocol
    return sanitize_audit_record(record, protocol=str(record["protocol"]))


def write_audit_json(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Write a sanitized JSON audit record and return its resolved path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_audit_record(record, protocol=str(record.get("protocol", "abstract")))
    with target.open("w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return target.resolve()
