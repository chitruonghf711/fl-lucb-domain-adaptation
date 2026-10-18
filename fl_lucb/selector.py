"""Budget-bounded LUCB selection over pre-planned minibatches."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from .audit import build_audit_record
from .minibatch import Minibatch, MinibatchPlan
from .models import (
    ArmEvidence,
    BatchScore,
    ClassAssessment,
    ScoredBatch,
    SelectionConfig,
    SelectionResult,
    SelectionState,
    WilsonInterval,
    label_key,
)
from .wilson import error_interval, wilson_interval


ScoreCallback = Callable[[str, Minibatch], Any]


class SelectionInputError(ValueError):
    """Raised internally when plans or scorer output violate the safe contract."""


def _normalize_plans(
    plans: Mapping[str, MinibatchPlan] | Sequence[MinibatchPlan],
) -> Dict[str, MinibatchPlan]:
    if isinstance(plans, Mapping):
        candidate = dict(plans)
    else:
        candidate = {plan.client_id: plan for plan in plans}
        if len(candidate) != len(plans):
            raise SelectionInputError("plans must have unique client IDs")
    if not candidate:
        raise SelectionInputError("at least one client plan is required")
    normalized: Dict[str, MinibatchPlan] = {}
    for client_id, plan in candidate.items():
        if not isinstance(plan, MinibatchPlan):
            raise SelectionInputError("plans must contain MinibatchPlan values")
        if client_id != plan.client_id:
            raise SelectionInputError("plan mapping keys must match plan.client_id")
        normalized[client_id] = plan
    return dict(sorted(normalized.items()))


def _normalize_score(value: Any, batch: Minibatch) -> BatchScore:
    """Normalize accepted scorer output and verify it describes this batch only."""

    if isinstance(value, BatchScore):
        score = value
    elif isinstance(value, Mapping):
        if "predictions" in value:
            score = BatchScore.from_predictions(batch.labels, value["predictions"])
        elif "correctness" in value:
            score = BatchScore.from_correctness(batch.labels, value["correctness"])
        elif {"errors", "total", "correct_by_label", "total_by_label"}.issubset(value):
            score = BatchScore(
                errors=value["errors"],
                total=value["total"],
                correct_by_label=value["correct_by_label"],
                total_by_label=value["total_by_label"],
                correctness=tuple(value.get("correctness", ())),
            )
        else:
            raise SelectionInputError("scorer mapping must contain predictions, correctness, or aggregate score fields")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        score = BatchScore.from_predictions(batch.labels, value)
    else:
        raise SelectionInputError("scorer must return BatchScore, a score mapping, or ordered predictions")

    expected_totals = Counter(batch.labels)
    if score.total != batch.size:
        raise SelectionInputError("scorer must score every sample in the supplied minibatch exactly once")
    if dict(score.total_by_label) != dict(expected_totals):
        raise SelectionInputError("scorer label totals do not match the supplied minibatch")
    return score


class LUCBSelector:
    """Select high-error clients without evaluating samples outside supplied plans.

    The selector treats error rate as the arm score: larger error means a weaker
    client. It samples an initially unobserved batch from every client, then
    samples the uncertain inner and outer boundary arms until the confidence
    boundary is separated within ``lucb_epsilon`` or the budget is exhausted.
    """

    def __init__(self, config: SelectionConfig | None = None):
        self.config = config or SelectionConfig()

    def select(
        self,
        plans: Mapping[str, MinibatchPlan] | Sequence[MinibatchPlan],
        scorer: ScoreCallback,
        *,
        model_fingerprint: str,
        state: SelectionState | None = None,
        budget: int | None = None,
        event_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SelectionResult:
        """Run one safe selection event.

        The callback is invoked only with a ``Minibatch`` from ``plans``. The
        reported model-sample cost is the number of callback-scored samples and
        never exceeds ``budget``. Invalid input and insufficient warm-up budget
        return explicit result statuses without calling the scorer.
        """

        safe_fingerprint = model_fingerprint if isinstance(model_fingerprint, str) else ""
        try:
            self.config.validate()
            if not safe_fingerprint:
                raise SelectionInputError("model_fingerprint must be a non-empty string")
            normalized_plans = _normalize_plans(plans)
            if not callable(scorer):
                raise SelectionInputError("scorer must be callable")
            if state is not None and not isinstance(state, SelectionState):
                raise SelectionInputError("state must be a SelectionState")
        except (ValueError, SelectionInputError) as exc:
            return self._empty_result(
                status="invalid",
                stop_reason="invalid_input",
                model_fingerprint=safe_fingerprint,
                budget=0,
                event_id=event_id,
                metadata={"error_type": type(exc).__name__},
            )

        state = state or SelectionState()
        total_capacity = sum(plan.total_samples for plan in normalized_plans.values())
        try:
            resolved_budget = (
                total_capacity
                if self.config.selection_mode == "legacy_full_scan" and budget is None
                else self._resolve_budget(total_capacity, budget)
            )
        except ValueError as exc:
            return self._empty_result(
                status="invalid",
                stop_reason="invalid_budget",
                model_fingerprint=safe_fingerprint,
                budget=0,
                event_id=event_id,
                metadata={"error_type": type(exc).__name__},
                plans=normalized_plans,
            )

        self._prepare_state(state, safe_fingerprint, normalized_plans)
        if any(not plan.batches for plan in normalized_plans.values()):
            return self._result(
                status="insufficient",
                stop_reason="empty_client_plan",
                model_fingerprint=safe_fingerprint,
                state=state,
                plans=normalized_plans,
                budget=resolved_budget,
                event_id=event_id,
                metadata=metadata,
            )

        if self.config.selection_mode == "legacy_full_scan":
            return self._select_legacy(
                normalized_plans,
                scorer,
                safe_fingerprint,
                state,
                resolved_budget,
                event_id,
                metadata,
            )
        return self._select_lucb(
            normalized_plans,
            scorer,
            safe_fingerprint,
            state,
            resolved_budget,
            event_id,
            metadata,
        )

    def _resolve_budget(self, total_capacity: int, budget: int | None) -> int:
        if total_capacity < 0:
            raise ValueError("total capacity cannot be negative")
        if budget is None:
            return int(total_capacity * self.config.lucb_budget_fraction)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative integer")
        return min(budget, total_capacity)

    def _prepare_state(
        self,
        state: SelectionState,
        model_fingerprint: str,
        plans: Mapping[str, MinibatchPlan],
    ) -> None:
        """Reset evidence for a changed model or altered client plan."""

        if state.model_fingerprint != model_fingerprint:
            state.reset_evidence(model_fingerprint)
        for client_id, plan in plans.items():
            prior_plan = state.plan_fingerprints.get(client_id)
            if prior_plan is not None and prior_plan != plan.fingerprint:
                state.evidence_by_client.pop(client_id, None)
            state.plan_fingerprints[client_id] = plan.fingerprint
            state.evidence_by_client.setdefault(client_id, ArmEvidence())
        removed_clients = set(state.evidence_by_client).difference(plans)
        for client_id in removed_clients:
            state.evidence_by_client.pop(client_id, None)
            state.plan_fingerprints.pop(client_id, None)

    def _current_cost(self, state: SelectionState, plans: Mapping[str, MinibatchPlan]) -> int:
        return sum(state.evidence_by_client[client_id].total for client_id in plans)

    def _confidence_allocation(self, plans: Mapping[str, MinibatchPlan]) -> Dict[str, Any]:
        """Allocate the configured error budget over finite planned looks.

        A plan has a finite number of evidence updates. Applying a Wilson
        interval after each update therefore needs a conservative union bound;
        using the raw ``lucb_delta`` at every adaptive look would overstate its
        coverage. The allocation also reserves budget for class-level
        assessments, so feedback classifications share the same error budget.
        """

        error_look_count = sum(len(plan.batches) for plan in plans.values())
        class_look_count = sum(
            len(plan.batches) * len({sample.label for batch in plan.batches for sample in batch.samples})
            for plan in plans.values()
        )
        total_look_count = error_look_count + class_look_count
        effective_delta = (
            self.config.lucb_delta / total_look_count
            if total_look_count > 0
            else self.config.lucb_delta
        )
        return {
            "method": "finite_horizon_bonferroni",
            "base_lucb_delta": self.config.lucb_delta,
            "error_interval_look_count": error_look_count,
            "class_interval_look_count": class_look_count,
            "effective_error_interval_delta": effective_delta,
            "effective_class_interval_delta": effective_delta,
        }

    def _ordered_client_ids(
        self,
        client_ids: Iterable[str],
        state: SelectionState,
        evidence: Mapping[str, ArmEvidence] | None = None,
    ) -> list[str]:
        """Rank arms by error evidence; optional priority breaks only ties."""

        evidence = evidence or state.evidence_by_client

        def key(client_id: str) -> tuple[float, float, str]:
            arm = evidence[client_id]
            priority = state.priority_by_client.get(client_id, 0.0)
            priority_key = -priority if self.config.lucb_prioritization == "decay_priority" else 0.0
            return (-arm.error_rate, priority_key, client_id)

        return sorted(client_ids, key=key)

    def _warmup_cost(self, plans: Mapping[str, MinibatchPlan], state: SelectionState) -> int:
        required = 0
        for client_id, plan in plans.items():
            evidence = state.evidence_by_client[client_id]
            if evidence.total == 0:
                first = plan.next_unscored(evidence.scored_batch_indices)
                if first is None:
                    return -1
                required += first.size
        return required

    def _select_lucb(
        self,
        plans: Mapping[str, MinibatchPlan],
        scorer: ScoreCallback,
        model_fingerprint: str,
        state: SelectionState,
        budget: int,
        event_id: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> SelectionResult:
        error_delta = self._confidence_allocation(plans)["effective_error_interval_delta"]
        current_cost = self._current_cost(state, plans)
        if current_cost > budget:
            return self._empty_result(
                status="invalid",
                stop_reason="state_exceeds_budget",
                model_fingerprint=model_fingerprint,
                plans=plans,
                budget=budget,
                event_id=event_id,
                metadata=metadata,
            )
        warmup_cost = self._warmup_cost(plans, state)
        if warmup_cost < 0:
            return self._result(
                status="insufficient",
                stop_reason="no_unscored_batch_for_warmup",
                model_fingerprint=model_fingerprint,
                state=state,
                plans=plans,
                budget=budget,
                event_id=event_id,
                metadata=metadata,
            )
        if current_cost + warmup_cost > budget:
            return self._result(
                status="insufficient",
                stop_reason="insufficient_budget_for_warmup",
                model_fingerprint=model_fingerprint,
                state=state,
                plans=plans,
                budget=budget,
                event_id=event_id,
                metadata=metadata,
            )

        for client_id in self._ordered_client_ids(plans, state):
            evidence = state.evidence_by_client[client_id]
            if evidence.total == 0:
                batch = plans[client_id].next_unscored(evidence.scored_batch_indices)
                assert batch is not None
                failure = self._score_one(client_id, batch, scorer, state)
                if failure is not None:
                    return self._result(
                        status="invalid",
                        stop_reason="scoring_error",
                        model_fingerprint=model_fingerprint,
                        state=state,
                        plans=plans,
                        budget=budget,
                        event_id=event_id,
                        metadata={"error_type": failure},
                    )

        client_ids = list(plans)
        top_k = self.config.resolved_top_k(len(client_ids))
        if top_k == len(client_ids):
            self._update_priority(state, client_ids)
            return self._result(
                status="ok",
                stop_reason="all_arms_selected",
                model_fingerprint=model_fingerprint,
                state=state,
                plans=plans,
                budget=budget,
                event_id=event_id,
                metadata=metadata,
            )

        while True:
            order = self._ordered_client_ids(client_ids, state)
            inside = order[:top_k]
            outside = order[top_k:]
            intervals = {
                client_id: error_interval(
                    state.evidence_by_client[client_id].errors,
                    state.evidence_by_client[client_id].total,
                    error_delta,
                )
                for client_id in client_ids
            }
            inner = min(inside, key=lambda client_id: (intervals[client_id].lower, client_id))
            outer = max(outside, key=lambda client_id: (intervals[client_id].upper, client_id))
            boundary_gap = intervals[inner].lower - intervals[outer].upper
            if boundary_gap >= -self.config.lucb_epsilon:
                self._update_priority(state, client_ids)
                return self._result(
                    status="ok",
                    stop_reason="confidence_separated" if boundary_gap >= 0 else "epsilon_reached",
                    model_fingerprint=model_fingerprint,
                    state=state,
                    plans=plans,
                    budget=budget,
                    event_id=event_id,
                    metadata=metadata,
                    boundary_gap=boundary_gap,
                )

            current_cost = self._current_cost(state, plans)
            candidates = [inner, outer]
            # Repeating one arm in a tie would waste budget and violate the
            # intended two-boundary exploration policy.
            candidates = list(dict.fromkeys(candidates))
            scored = False
            for client_id in candidates:
                batch = plans[client_id].next_unscored(
                    state.evidence_by_client[client_id].scored_batch_indices
                )
                if batch is None or current_cost + batch.size > budget:
                    continue
                failure = self._score_one(client_id, batch, scorer, state)
                if failure is not None:
                    return self._result(
                        status="invalid",
                        stop_reason="scoring_error",
                        model_fingerprint=model_fingerprint,
                        state=state,
                        plans=plans,
                        budget=budget,
                        event_id=event_id,
                        metadata={"error_type": failure},
                        boundary_gap=boundary_gap,
                    )
                scored = True
                break
            if scored:
                continue

            all_exhausted = all(
                plan.next_unscored(state.evidence_by_client[client_id].scored_batch_indices) is None
                for client_id, plan in plans.items()
            )
            self._update_priority(state, client_ids)
            return self._result(
                status="ok" if all_exhausted else "partial",
                stop_reason="all_batches_exhausted" if all_exhausted else "budget_exhausted",
                model_fingerprint=model_fingerprint,
                state=state,
                plans=plans,
                budget=budget,
                event_id=event_id,
                metadata=metadata,
                boundary_gap=boundary_gap,
            )

    def _select_legacy(
        self,
        plans: Mapping[str, MinibatchPlan],
        scorer: ScoreCallback,
        model_fingerprint: str,
        state: SelectionState,
        budget: int,
        event_id: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> SelectionResult:
        """Run the explicit baseline only when every planned sample fits budget."""

        current_cost = self._current_cost(state, plans)
        remaining_cost = sum(
            batch.size
            for client_id, plan in plans.items()
            for batch in plan.batches
            if batch.batch_index not in state.evidence_by_client[client_id].scored_batch_indices
        )
        if current_cost + remaining_cost > budget:
            return self._result(
                status="insufficient",
                stop_reason="insufficient_budget_for_legacy_full_scan",
                model_fingerprint=model_fingerprint,
                state=state,
                plans=plans,
                budget=budget,
                event_id=event_id,
                metadata=metadata,
            )
        for client_id in self._ordered_client_ids(plans, state):
            plan = plans[client_id]
            for batch in plan.batches:
                if batch.batch_index in state.evidence_by_client[client_id].scored_batch_indices:
                    continue
                failure = self._score_one(client_id, batch, scorer, state)
                if failure is not None:
                    return self._result(
                        status="invalid",
                        stop_reason="scoring_error",
                        model_fingerprint=model_fingerprint,
                        state=state,
                        plans=plans,
                        budget=budget,
                        event_id=event_id,
                        metadata={"error_type": failure},
                    )
        self._update_priority(state, plans)
        return self._result(
            status="ok",
            stop_reason="legacy_full_scan_complete",
            model_fingerprint=model_fingerprint,
            state=state,
            plans=plans,
            budget=budget,
            event_id=event_id,
            metadata=metadata,
        )

    def _score_one(
        self,
        client_id: str,
        batch: Minibatch,
        scorer: ScoreCallback,
        state: SelectionState,
    ) -> str | None:
        try:
            score = _normalize_score(scorer(client_id, batch), batch)
            scored_batch = ScoredBatch(
                client_id=client_id,
                batch_index=batch.batch_index,
                sample_ids=batch.sample_ids,
                labels=batch.labels,
                correctness=score.correctness,
            )
            state.evidence_by_client[client_id].add(
                batch.batch_index,
                batch.sample_ids,
                score,
                scored_batch=scored_batch,
            )
            return None
        except Exception as exc:  # The result reports only the exception type.
            return type(exc).__name__

    def _update_priority(self, state: SelectionState, client_ids: Iterable[str]) -> None:
        """Update scheduling priority without adding historical statistical evidence."""

        if self.config.lucb_prioritization != "decay_priority":
            return
        for client_id in client_ids:
            evidence = state.evidence_by_client[client_id]
            previous = state.priority_by_client.get(client_id, evidence.error_rate)
            state.priority_by_client[client_id] = (
                self.config.priority_decay * previous
                + (1 - self.config.priority_decay) * evidence.error_rate
            )

    def _assess_classes(
        self, evidence: ArmEvidence, class_interval_delta: float
    ) -> tuple[ClassAssessment, ...]:
        assessments = []
        for label in sorted(evidence.total_by_label, key=label_key):
            total = evidence.total_by_label[label]
            correct = evidence.correct_by_label.get(label, 0)
            interval = wilson_interval(correct, total, class_interval_delta)
            if interval.upper < self.config.weak_class_accuracy:
                status = "weak"
            elif interval.lower > self.config.weak_class_accuracy:
                status = "adequate"
            else:
                status = "uncertain"
            assessments.append(
                ClassAssessment(
                    label=label,
                    accuracy=interval.estimate,
                    lower=interval.lower,
                    upper=interval.upper,
                    total=total,
                    correct=correct,
                    status=status,
                )
            )
        return tuple(assessments)

    def _result(
        self,
        *,
        status: str,
        stop_reason: str,
        model_fingerprint: str,
        state: SelectionState,
        plans: Mapping[str, MinibatchPlan],
        budget: int,
        event_id: str | None,
        metadata: Mapping[str, Any] | None,
        boundary_gap: float | None = None,
    ) -> SelectionResult:
        client_ids = list(plans)
        confidence_allocation = self._confidence_allocation(plans)
        top_k = self.config.resolved_top_k(len(client_ids))
        selected = (
            ()
            if status in {"invalid", "insufficient"}
            else tuple(self._ordered_client_ids(client_ids, state)[:top_k])
        )
        class_assessments = {
            client_id: self._assess_classes(
                state.evidence_by_client[client_id],
                confidence_allocation["effective_class_interval_delta"],
            )
            for client_id in client_ids
        }
        selected_classes = {
            client_id: tuple(
                assessment
                for assessment in class_assessments[client_id]
                if assessment.status == "weak"
            )
            for client_id in selected
        }
        confidence_bounds = {
            client_id: error_interval(
                state.evidence_by_client[client_id].errors,
                state.evidence_by_client[client_id].total,
                confidence_allocation["effective_error_interval_delta"],
            )
            for client_id in client_ids
        }
        scored_sample_ids = {
            client_id: tuple(state.evidence_by_client[client_id].scored_sample_ids)
            for client_id in client_ids
        }
        scored_batches = {
            client_id: tuple(
                state.evidence_by_client[client_id].scored_batches[index]
                for index in sorted(state.evidence_by_client[client_id].scored_batches)
            )
            for client_id in client_ids
        }
        coverage = {
            client_id: {
                **plan.coverage_summary(),
                "scored_sample_count": state.evidence_by_client[client_id].total,
                "scored_batch_indices": sorted(state.evidence_by_client[client_id].scored_batch_indices),
            }
            for client_id, plan in plans.items()
        }
        cost = self._current_cost(state, plans)
        audit = build_audit_record(
            event_id=event_id,
            configuration={
                **self.config.to_dict(),
                "confidence_allocation": confidence_allocation,
            },
            model_fingerprint=model_fingerprint,
            status=status,
            stop_reason=stop_reason,
            model_sample_evaluations=min(cost, budget),
            budget=budget,
            selected_client_ids=selected,
            selected_classes=selected_classes,
            confidence_bounds=confidence_bounds,
            coverage=coverage,
            scored_batches=scored_batches,
            boundary_gap=boundary_gap,
            protocol="abstract",
            metadata={
                **dict(metadata or {}),
                "confidence_allocation": confidence_allocation,
            },
        )
        return SelectionResult(
            status=status,
            stop_reason=stop_reason,
            model_fingerprint=model_fingerprint,
            selected_client_ids=selected,
            selected_classes=selected_classes,
            class_assessments=class_assessments,
            confidence_bounds=confidence_bounds,
            scored_sample_ids=scored_sample_ids,
            scored_batches=scored_batches,
            coverage=coverage,
            model_sample_evaluations=cost,
            budget=budget,
            boundary_gap=boundary_gap,
            audit=audit,
        )

    def _empty_result(
        self,
        *,
        status: str,
        stop_reason: str,
        model_fingerprint: str,
        budget: int,
        event_id: str | None,
        metadata: Mapping[str, Any] | None,
        plans: Mapping[str, MinibatchPlan] | None = None,
    ) -> SelectionResult:
        plans = plans or {}
        coverage = {client_id: plan.coverage_summary() for client_id, plan in plans.items()}
        audit = build_audit_record(
            event_id=event_id,
            configuration=self.config.to_dict(),
            model_fingerprint=model_fingerprint,
            status=status,
            stop_reason=stop_reason,
            model_sample_evaluations=0,
            budget=budget,
            selected_client_ids=(),
            selected_classes={},
            confidence_bounds={},
            coverage=coverage,
            scored_batches={},
            boundary_gap=None,
            protocol="abstract",
            metadata=metadata,
        )
        return SelectionResult(
            status=status,
            stop_reason=stop_reason,
            model_fingerprint=model_fingerprint,
            selected_client_ids=(),
            selected_classes={},
            class_assessments={},
            confidence_bounds={},
            scored_sample_ids={},
            scored_batches={},
            coverage=coverage,
            model_sample_evaluations=0,
            budget=budget,
            boundary_gap=None,
            audit=audit,
        )
