"""Dependency-free minibatch LUCB selection for federated feedback workflows."""

from .audit import build_audit_record, sanitize_audit_record, write_audit_json
from .minibatch import Minibatch, MinibatchPlan, SampleRef, build_minibatch_plans
from .models import (
    ArmEvidence,
    BatchScore,
    ClassAssessment,
    ScoredBatch,
    SelectionConfig,
    SelectionResult,
    SelectionState,
    WilsonInterval,
)
from .protocols import (
    FeedbackProtocolError,
    sanitize_feedback_for_audit,
    validate_feedback_payload,
)
from .selector import LUCBSelector, ScoreCallback, SelectionInputError
from .wilson import error_interval, wilson_interval

__all__ = [
    "ArmEvidence",
    "BatchScore",
    "ClassAssessment",
    "FeedbackProtocolError",
    "LUCBSelector",
    "Minibatch",
    "MinibatchPlan",
    "SampleRef",
    "ScoreCallback",
    "ScoredBatch",
    "SelectionConfig",
    "SelectionInputError",
    "SelectionResult",
    "SelectionState",
    "WilsonInterval",
    "build_audit_record",
    "build_minibatch_plans",
    "error_interval",
    "sanitize_audit_record",
    "sanitize_feedback_for_audit",
    "validate_feedback_payload",
    "wilson_interval",
    "write_audit_json",
]
