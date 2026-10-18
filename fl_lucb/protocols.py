"""Feedback protocol validation for privacy-preserving integrations."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence

from .audit import sanitize_audit_record
from .models import json_safe


class FeedbackProtocolError(ValueError):
    """Raised when a feedback payload violates its selected protocol."""


_DOMAINS = {"tabular", "text", "image"}
_COMMON_FIELDS = {
    "protocol",
    "domain",
    "weak_labels",
    "client_id",
    "selected_client_id",
    "event_id",
    "round",
    "model_fingerprint",
}
_RAW_FIELD_PARTS = (
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
)
_RAW_EXACT_FIELDS = {"rows", "image", "images", "values", "samples", "inputs"}


def _is_raw_field(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _RAW_EXACT_FIELDS or any(part in normalized for part in _RAW_FIELD_PARTS)


def _contains_raw_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _is_raw_field(key) or _contains_raw_field(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_raw_field(item) for item in value)
    return False


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeedbackProtocolError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise FeedbackProtocolError(f"{field} must be a list")
    return value


def _safe_scalar(value: Any, field: str) -> Any:
    safe = json_safe(value)
    if isinstance(safe, (dict, list)):
        raise FeedbackProtocolError(f"{field} must be a scalar")
    return safe


def _safe_number(value: Any, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeedbackProtocolError(f"{field} must be a number")
    number = float(value)
    if not isfinite(number):
        raise FeedbackProtocolError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise FeedbackProtocolError(f"{field} must be at least {minimum}")
    return number


def _normalized_common(
    payload: Mapping[str, Any], protocol: str, domain: str
) -> dict[str, Any]:
    weak_labels = _require_list(payload.get("weak_labels"), "weak_labels")
    normalized: dict[str, Any] = {
        "protocol": protocol,
        "domain": domain,
        "weak_labels": [_safe_scalar(label, "weak_labels item") for label in weak_labels],
    }
    for field in (
        "client_id",
        "selected_client_id",
        "event_id",
        "round",
        "model_fingerprint",
    ):
        if field in payload:
            normalized[field] = _safe_scalar(payload[field], field)
    return normalized


def _validate_tabular_abstract(
    payload: Mapping[str, Any], normalized: dict[str, Any]
) -> None:
    allowed = _COMMON_FIELDS | {"feature_ranking"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise FeedbackProtocolError(f"abstract tabular payload has unknown fields: {sorted(unknown)}")
    ranking = _require_list(payload.get("feature_ranking", []), "feature_ranking")
    cleaned = []
    for item in ranking:
        entry = _require_mapping(item, "feature_ranking item")
        if set(entry).difference({"feature", "rank", "importance"}):
            raise FeedbackProtocolError("feature_ranking items only allow feature, rank, and importance")
        feature = entry.get("feature")
        if not isinstance(feature, str) or not feature:
            raise FeedbackProtocolError("feature_ranking feature must be a non-empty string")
        rank = entry.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise FeedbackProtocolError("feature_ranking rank must be a positive integer")
        cleaned.append(
            {
                "feature": feature,
                "rank": rank,
                "importance": _safe_number(entry.get("importance"), "feature_ranking importance", 0),
            }
        )
    normalized["feature_ranking"] = cleaned


def _validate_text_abstract(payload: Mapping[str, Any], normalized: dict[str, Any]) -> None:
    allowed = _COMMON_FIELDS | {"salient_positions", "position_distribution"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise FeedbackProtocolError(f"abstract text payload has unknown fields: {sorted(unknown)}")
    positions = _require_list(payload.get("salient_positions", []), "salient_positions")
    cleaned_positions = []
    for item in positions:
        entry = _require_mapping(item, "salient_positions item")
        if set(entry).difference({"position", "weight"}):
            raise FeedbackProtocolError("salient_positions items only allow position and weight")
        position = entry.get("position")
        if position not in {"beginning", "middle", "end"}:
            raise FeedbackProtocolError("salient position must be beginning, middle, or end")
        cleaned_positions.append(
            {"position": position, "weight": _safe_number(entry.get("weight"), "position weight", 0)}
        )
    normalized["salient_positions"] = cleaned_positions
    if "position_distribution" in payload:
        distribution = _require_mapping(payload["position_distribution"], "position_distribution")
        if set(distribution).difference({"beginning", "middle", "end"}):
            raise FeedbackProtocolError("position_distribution has an invalid position")
        normalized["position_distribution"] = {
            position: _safe_number(weight, "position distribution weight", 0)
            for position, weight in distribution.items()
        }


def _validate_image_abstract(payload: Mapping[str, Any], normalized: dict[str, Any]) -> None:
    allowed = _COMMON_FIELDS | {"generation_mode", "patch_grid_size", "salient_patches"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise FeedbackProtocolError(f"abstract image payload has unknown fields: {sorted(unknown)}")
    generation_mode = payload.get("generation_mode", "refine")
    if generation_mode != "refine":
        raise FeedbackProtocolError("abstract image feedback only supports generation_mode='refine'")
    normalized["generation_mode"] = "refine"
    if "patch_grid_size" in payload:
        grid_size = payload["patch_grid_size"]
        if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size <= 0:
            raise FeedbackProtocolError("patch_grid_size must be a positive integer")
        normalized["patch_grid_size"] = grid_size
    patches = _require_list(payload.get("salient_patches", []), "salient_patches")
    cleaned_patches = []
    for item in patches:
        entry = _require_mapping(item, "salient_patches item")
        if set(entry).difference({"row", "column", "weight"}):
            raise FeedbackProtocolError("salient_patches items only allow row, column, and weight")
        row = entry.get("row")
        column = entry.get("column")
        if (
            not isinstance(row, int)
            or isinstance(row, bool)
            or row < 0
            or not isinstance(column, int)
            or isinstance(column, bool)
            or column < 0
        ):
            raise FeedbackProtocolError("salient patch row and column must be non-negative integers")
        cleaned_patches.append(
            {"row": row, "column": column, "weight": _safe_number(entry.get("weight"), "patch weight", 0)}
        )
    normalized["salient_patches"] = cleaned_patches


def validate_feedback_payload(
    payload: Mapping[str, Any],
    *,
    protocol: str | None = None,
    domain: str | None = None,
    allow_raw_feedback: bool = False,
) -> dict[str, Any]:
    """Validate and normalize an abstract or explicitly enabled legacy payload.

    Abstract payloads intentionally accept only labels and aggregate explanation
    metadata. Legacy payloads are permitted only with an explicit opt-in and
    should never be copied into an audit record.
    """

    if not isinstance(payload, Mapping):
        raise FeedbackProtocolError("feedback payload must be an object")
    effective_protocol = protocol or payload.get("protocol", "abstract")
    effective_domain = domain or payload.get("domain")
    if effective_protocol not in {"abstract", "legacy"}:
        raise FeedbackProtocolError("protocol must be 'abstract' or 'legacy'")
    if effective_domain not in _DOMAINS:
        raise FeedbackProtocolError("domain must be tabular, text, or image")
    if "protocol" in payload and payload["protocol"] != effective_protocol:
        raise FeedbackProtocolError("payload protocol conflicts with requested protocol")
    if "domain" in payload and payload["domain"] != effective_domain:
        raise FeedbackProtocolError("payload domain conflicts with requested domain")

    if effective_protocol == "legacy":
        if not allow_raw_feedback:
            raise FeedbackProtocolError("legacy feedback requires allow_raw_feedback=True")
        normalized = dict(payload)
        normalized["protocol"] = "legacy"
        normalized["domain"] = effective_domain
        _normalized_common(normalized, "legacy", effective_domain)
        return normalized

    if _contains_raw_field(payload):
        raise FeedbackProtocolError("abstract feedback cannot contain raw source content")
    normalized = _normalized_common(payload, "abstract", effective_domain)
    if effective_domain == "tabular":
        _validate_tabular_abstract(payload, normalized)
    elif effective_domain == "text":
        _validate_text_abstract(payload, normalized)
    else:
        _validate_image_abstract(payload, normalized)
    return normalized


def sanitize_feedback_for_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip raw fields from either protocol before embedding feedback in audit."""

    record = sanitize_audit_record({"feedback": payload}, protocol="abstract")
    return dict(record.get("feedback", {}))
