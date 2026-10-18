"""Data models for dependency-free LUCB client selection.

The models in this module deliberately contain metadata, labels, and aggregate
correctness only. They do not retain model inputs, text, pixels, or feature
values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, Hashable, Iterable, Mapping, Sequence, Tuple


Label = Hashable


def json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation of ``value``.

    Unknown objects are represented by their string form instead of exposing
    object internals. This keeps audit serialization safe for common numeric
    scalar types supplied by data libraries without importing those libraries.
    """

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=repr)]
    return str(value)


def label_key(label: Label) -> str:
    """Create a stable, type-aware key for labels in serialized records."""

    return f"{type(label).__name__}:{json_safe(label)!r}"


@dataclass(frozen=True)
class WilsonInterval:
    """A Wilson confidence interval for a Bernoulli success rate."""

    successes: int
    total: int
    estimate: float
    lower: float
    upper: float
    delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "successes": self.successes,
            "total": self.total,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class BatchScore:
    """Aggregate correctness for exactly one planned minibatch.

    ``errors`` counts incorrect predictions. ``correct_by_label`` and
    ``total_by_label`` allow per-class accuracy without storing predictions or
    input samples.
    """

    errors: int
    total: int
    correct_by_label: Mapping[Label, int]
    total_by_label: Mapping[Label, int]
    correctness: Tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if self.total < 0 or self.errors < 0 or self.errors > self.total:
            raise ValueError("BatchScore errors and total must be non-negative and consistent")
        if sum(self.total_by_label.values()) != self.total:
            raise ValueError("BatchScore label totals must equal total")
        if sum(self.correct_by_label.values()) != self.total - self.errors:
            raise ValueError("BatchScore label correctness must agree with errors")
        for label, total in self.total_by_label.items():
            if total < 0:
                raise ValueError("BatchScore label totals must be non-negative")
            correct = self.correct_by_label.get(label, 0)
            if correct < 0 or correct > total:
                raise ValueError("BatchScore label correctness must not exceed label total")
        extra_labels = set(self.correct_by_label).difference(self.total_by_label)
        if extra_labels:
            raise ValueError("BatchScore correctness contains unknown labels")
        if self.correctness:
            if len(self.correctness) != self.total:
                raise ValueError("BatchScore correctness must match total when supplied")
            if sum(bool(value) for value in self.correctness) != self.total - self.errors:
                raise ValueError("BatchScore correctness must agree with errors")

    @classmethod
    def from_predictions(
        cls, labels: Sequence[Label], predictions: Sequence[Label]
    ) -> "BatchScore":
        """Build a score by comparing ordered predictions to ordered labels."""

        if len(labels) != len(predictions):
            raise ValueError("Predictions must have the same length as labels")
        correctness = [label == prediction for label, prediction in zip(labels, predictions)]
        return cls.from_correctness(labels, correctness)

    @classmethod
    def from_correctness(
        cls, labels: Sequence[Label], correctness: Sequence[bool]
    ) -> "BatchScore":
        """Build a score from ordered boolean correctness flags."""

        if len(labels) != len(correctness):
            raise ValueError("Correctness flags must have the same length as labels")
        totals: Dict[Label, int] = {}
        correct: Dict[Label, int] = {}
        error_count = 0
        for label, is_correct in zip(labels, correctness):
            try:
                hash(label)
            except TypeError as exc:
                raise ValueError("Labels must be hashable") from exc
            totals[label] = totals.get(label, 0) + 1
            if bool(is_correct):
                correct[label] = correct.get(label, 0) + 1
            else:
                error_count += 1
        return cls(
            errors=error_count,
            total=len(labels),
            correct_by_label=correct,
            total_by_label=totals,
            correctness=tuple(bool(value) for value in correctness),
        )

@dataclass(frozen=True)
class ClassAssessment:
    """Confidence-aware class accuracy classification for one client."""

    label: Label
    accuracy: float
    lower: float
    upper: float
    total: int
    correct: int
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": json_safe(self.label),
            "accuracy": self.accuracy,
            "lower": self.lower,
            "upper": self.upper,
            "total": self.total,
            "correct": self.correct,
            "status": self.status,
        }


@dataclass(frozen=True)
class ScoredBatch:
    """An audit-safe record of a scored planned batch.

    ``correctness`` is optional because selectors do not need to retain it.
    Integrators should keep raw rows or model outputs locally and use
    ``sample_ids`` to limit explanations to this set.
    """

    client_id: str
    batch_index: int
    sample_ids: Tuple[Any, ...]
    labels: Tuple[Label, ...]
    correctness: Tuple[bool, ...] = ()

    def to_audit_dict(self) -> Dict[str, Any]:
        """Return metadata without labels or per-sample correctness."""

        return {
            "client_id": self.client_id,
            "batch_index": self.batch_index,
            "sample_ids": [json_safe(sample_id) for sample_id in self.sample_ids],
            "sample_count": len(self.sample_ids),
        }


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration shared by tabular, text, and image integrations."""

    selection_mode: str = "lucb"
    lucb_budget_fraction: float = 0.25
    lucb_batch_size: int = 64
    lucb_top_k: int | None = None
    lucb_epsilon: float = 0.02
    lucb_delta: float = 0.05
    weak_class_accuracy: float = 0.85
    seed: int = 42
    lucb_prioritization: str = "strict"
    priority_decay: float = 0.5

    def validate(self) -> None:
        if self.selection_mode not in {"lucb", "legacy_full_scan"}:
            raise ValueError("selection_mode must be 'lucb' or 'legacy_full_scan'")
        if not 0 < self.lucb_budget_fraction <= 1:
            raise ValueError("lucb_budget_fraction must be in (0, 1]")
        if self.lucb_batch_size <= 0:
            raise ValueError("lucb_batch_size must be positive")
        if self.lucb_top_k is not None and self.lucb_top_k <= 0:
            raise ValueError("lucb_top_k must be positive when provided")
        if self.lucb_epsilon < 0:
            raise ValueError("lucb_epsilon must be non-negative")
        if not 0 < self.lucb_delta < 1:
            raise ValueError("lucb_delta must be in (0, 1)")
        if not 0 <= self.weak_class_accuracy <= 1:
            raise ValueError("weak_class_accuracy must be in [0, 1]")
        if self.lucb_prioritization not in {"strict", "decay_priority"}:
            raise ValueError("lucb_prioritization must be 'strict' or 'decay_priority'")
        if not 0 <= self.priority_decay <= 1:
            raise ValueError("priority_decay must be in [0, 1]")

    def resolved_top_k(self, client_count: int) -> int:
        if client_count <= 0:
            return 0
        configured = 3 if self.lucb_top_k is None else self.lucb_top_k
        return min(configured, client_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_mode": self.selection_mode,
            "lucb_budget_fraction": self.lucb_budget_fraction,
            "lucb_batch_size": self.lucb_batch_size,
            "lucb_top_k": self.lucb_top_k,
            "lucb_epsilon": self.lucb_epsilon,
            "lucb_delta": self.lucb_delta,
            "weak_class_accuracy": self.weak_class_accuracy,
            "seed": self.seed,
            "lucb_prioritization": self.lucb_prioritization,
            "priority_decay": self.priority_decay,
        }


@dataclass
class ArmEvidence:
    """Cumulative same-model evidence for a single client arm."""

    errors: int = 0
    total: int = 0
    correct_by_label: Dict[Label, int] = field(default_factory=dict)
    total_by_label: Dict[Label, int] = field(default_factory=dict)
    scored_batch_indices: set[int] = field(default_factory=set)
    scored_sample_ids: list[Any] = field(default_factory=list)
    scored_batches: Dict[int, ScoredBatch] = field(default_factory=dict)

    def add(
        self,
        batch_index: int,
        sample_ids: Sequence[Any],
        score: BatchScore,
        scored_batch: ScoredBatch | None = None,
    ) -> None:
        """Add one non-overlapping scored batch after caller validation."""

        if batch_index in self.scored_batch_indices:
            raise ValueError("A minibatch cannot contribute evidence twice")
        self.scored_batch_indices.add(batch_index)
        self.scored_sample_ids.extend(sample_ids)
        if scored_batch is not None:
            self.scored_batches[batch_index] = scored_batch
        self.errors += score.errors
        self.total += score.total
        for label, count in score.correct_by_label.items():
            self.correct_by_label[label] = self.correct_by_label.get(label, 0) + count
        for label, count in score.total_by_label.items():
            self.total_by_label[label] = self.total_by_label.get(label, 0) + count

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total else 0.0


@dataclass
class SelectionState:
    """Mutable state that may be retained across selections.

    Evidence is valid only for the current model fingerprint. Priority is kept
    separately so optional decay scheduling can never alter Wilson evidence.
    """

    model_fingerprint: str | None = None
    evidence_by_client: Dict[str, ArmEvidence] = field(default_factory=dict)
    plan_fingerprints: Dict[str, str] = field(default_factory=dict)
    priority_by_client: Dict[str, float] = field(default_factory=dict)
    generation: int = 0

    def reset_evidence(self, model_fingerprint: str) -> None:
        """Clear all statistical evidence for a changed model."""

        self.model_fingerprint = model_fingerprint
        self.evidence_by_client.clear()
        self.plan_fingerprints.clear()
        self.generation += 1


@dataclass(frozen=True)
class SelectionResult:
    """Serializable result of a LUCB or legacy full-scan selection."""

    status: str
    stop_reason: str
    model_fingerprint: str
    selected_client_ids: Tuple[str, ...]
    selected_classes: Mapping[str, Tuple[ClassAssessment, ...]]
    class_assessments: Mapping[str, Tuple[ClassAssessment, ...]]
    confidence_bounds: Mapping[str, WilsonInterval]
    scored_sample_ids: Mapping[str, Tuple[Any, ...]]
    scored_batches: Mapping[str, Tuple[ScoredBatch, ...]]
    coverage: Mapping[str, Mapping[str, Any]]
    model_sample_evaluations: int
    budget: int
    boundary_gap: float | None
    audit: Mapping[str, Any]

    @property
    def feedback_ready(self) -> bool:
        """True only for a completed selection with a confident weak class."""

        return self.status == "ok" and any(self.selected_classes.values())

    def to_dict(self) -> Dict[str, Any]:
        """Return the result in a JSON-compatible form."""

        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "model_fingerprint": self.model_fingerprint,
            "selected_client_ids": list(self.selected_client_ids),
            "selected_classes": {
                client_id: [assessment.to_dict() for assessment in assessments]
                for client_id, assessments in self.selected_classes.items()
            },
            "class_assessments": {
                client_id: [assessment.to_dict() for assessment in assessments]
                for client_id, assessments in self.class_assessments.items()
            },
            "confidence_bounds": {
                client_id: interval.to_dict()
                for client_id, interval in self.confidence_bounds.items()
            },
            "scored_sample_ids": {
                client_id: [json_safe(sample_id) for sample_id in sample_ids]
                for client_id, sample_ids in self.scored_sample_ids.items()
            },
            "scored_batches": {
                client_id: [batch.to_audit_dict() for batch in batches]
                for client_id, batches in self.scored_batches.items()
            },
            "coverage": json_safe(self.coverage),
            "model_sample_evaluations": self.model_sample_evaluations,
            "budget": self.budget,
            "boundary_gap": self.boundary_gap,
            "audit": json_safe(self.audit),
        }
