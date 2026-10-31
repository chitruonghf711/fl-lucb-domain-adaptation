"""Pure standard-library tests for the shared LUCB package."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fl_lucb import (
    BatchScore,
    FeedbackProtocolError,
    LUCBSelector,
    MinibatchPlan,
    SelectionConfig,
    SelectionState,
    build_audit_record,
    sanitize_audit_record,
    validate_feedback_payload,
    wilson_interval,
    write_audit_json,
)


def make_plans(client_count: int = 5, sample_count: int = 12, batch_size: int = 2):
    return {
        f"client-{index}": MinibatchPlan.build(
            client_id=f"client-{index}",
            labels=[item % 2 for item in range(sample_count)],
            sample_ids=[f"{index}-{item}" for item in range(sample_count)],
            batch_size=batch_size,
            seed=42,
        )
        for index in range(client_count)
    }


class WilsonIntervalTests(unittest.TestCase):
    def test_wilson_bounds_and_empty_evidence(self):
        interval = wilson_interval(8, 10, 0.05)
        self.assertAlmostEqual(interval.estimate, 0.8)
        self.assertLess(interval.lower, 0.8)
        self.assertGreater(interval.upper, 0.8)

        empty = wilson_interval(0, 0, 0.05)
        self.assertEqual((empty.lower, empty.upper), (0.0, 1.0))

    def test_wilson_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)
        with self.assertRaises(ValueError):
            wilson_interval(0, 1, 1.0)


class MinibatchPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_and_non_repeating(self):
        labels = ["major"] * 11 + ["rare"]
        ids = [f"id-{index}" for index in range(len(labels))]
        first = MinibatchPlan.build("client-a", labels, 3, 91, ids)
        second = MinibatchPlan.build("client-a", labels, 3, 91, ids)

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.sample_ids, second.sample_ids)
        self.assertEqual(set(first.sample_ids), set(ids))
        self.assertEqual(len(first.sample_ids), len(set(first.sample_ids)))
        self.assertEqual(sum(len(batch.samples) for batch in first.batches), len(ids))

        observed_labels = [label for batch in first.batches for label in batch.labels]
        self.assertEqual(observed_labels.count("rare"), 1)
        self.assertLessEqual(max(len(batch.samples) for batch in first.batches), 3)
        self.assertGreater(len(first.coverage_summary()["batches"]), 1)

    def test_plan_rejects_duplicate_sample_ids(self):
        with self.assertRaises(ValueError):
            MinibatchPlan.build("client-a", [0, 1], 2, 1, ["same", "same"])


class SelectorTests(unittest.TestCase):
    def test_budget_never_exceeds_and_callback_only_receives_planned_batches(self):
        plans = make_plans()
        seen = []
        allowed = {
            client_id: {batch.batch_index for batch in plan.batches}
            for client_id, plan in plans.items()
        }
        error_clients = {"client-0", "client-3"}

        def scorer(client_id, batch):
            self.assertIn(batch.batch_index, allowed[client_id])
            seen.append((client_id, batch.batch_index, tuple(batch.sample_ids)))
            correctness = [client_id not in error_clients] * batch.size
            return BatchScore.from_correctness(batch.labels, correctness)

        config = SelectionConfig(lucb_batch_size=2, lucb_top_k=2, lucb_budget_fraction=1.0)
        result = LUCBSelector(config).select(
            plans,
            scorer,
            model_fingerprint="model-one",
            budget=12,
        )

        self.assertLessEqual(result.model_sample_evaluations, 12)
        self.assertEqual(sum(len(sample_ids) for _, _, sample_ids in seen), result.model_sample_evaluations)
        self.assertTrue(all(len(sample_ids) == 2 for _, _, sample_ids in seen))
        self.assertIn(result.stop_reason, {"budget_exhausted", "confidence_separated", "epsilon_reached"})
        self.assertEqual(set(result.selected_client_ids), {"client-0", "client-3"})

    def test_confidence_budget_is_allocated_over_all_planned_looks(self):
        plans = make_plans(client_count=2, sample_count=4, batch_size=2)
        config = SelectionConfig(
            lucb_batch_size=2,
            lucb_top_k=1,
            lucb_budget_fraction=1.0,
            lucb_delta=0.12,
        )
        result = LUCBSelector(config).select(
            plans,
            lambda client_id, batch: BatchScore.from_correctness(
                batch.labels, [client_id == "client-0"] * batch.size
            ),
            model_fingerprint="confidence-allocation",
            budget=8,
        )

        allocation = result.audit["configuration"]["confidence_allocation"]
        self.assertEqual(allocation["method"], "finite_horizon_bonferroni")
        self.assertGreater(allocation["error_interval_look_count"], 1)
        self.assertGreater(allocation["class_interval_look_count"], 1)
        self.assertLess(allocation["effective_error_interval_delta"], config.lucb_delta)
        self.assertTrue(
            all(
                interval.delta == allocation["effective_error_interval_delta"]
                for interval in result.confidence_bounds.values()
            )
        )

    def test_selected_class_feedback_candidates_are_only_confidently_weak(self):
        labels = [0] * 128
        plans = {
            "weak": MinibatchPlan.build("weak", labels, batch_size=64, seed=42),
            "adequate": MinibatchPlan.build("adequate", labels, batch_size=64, seed=42),
        }
        result = LUCBSelector(
            SelectionConfig(
                lucb_batch_size=64,
                lucb_top_k=1,
                lucb_budget_fraction=1.0,
                lucb_delta=0.5,
                weak_class_accuracy=0.85,
            )
        ).select(
            plans,
            lambda client_id, batch: BatchScore.from_correctness(
                batch.labels, [client_id == "adequate"] * batch.size
            ),
            model_fingerprint="weak-class-filter",
            budget=256,
        )

        self.assertEqual(result.selected_client_ids, ("weak",))
        self.assertTrue(result.selected_classes["weak"])
        self.assertTrue(
            all(item.status == "weak" for item in result.selected_classes["weak"])
        )
        self.assertTrue(
            all(item.status != "weak" for item in result.class_assessments["adequate"])
        )

    def test_insufficient_warmup_returns_no_selection_or_scoring(self):
        plans = make_plans(client_count=3, sample_count=4, batch_size=2)
        calls = []

        def scorer(client_id, batch):
            calls.append((client_id, batch.batch_index))
            return BatchScore.from_correctness(batch.labels, [True] * batch.size)

        result = LUCBSelector(SelectionConfig(lucb_batch_size=2)).select(
            plans,
            scorer,
            model_fingerprint="model-one",
            budget=5,
        )

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.stop_reason, "insufficient_budget_for_warmup")
        self.assertEqual(result.selected_client_ids, ())
        self.assertEqual(calls, [])
        self.assertEqual(result.model_sample_evaluations, 0)

    def test_budget_exhaustion_never_marks_feedback_ready(self):
        plans = make_plans(client_count=3, sample_count=4, batch_size=2)
        result = LUCBSelector(
            SelectionConfig(lucb_batch_size=2, lucb_top_k=1, lucb_budget_fraction=1.0)
        ).select(
            plans,
            lambda _client_id, batch: BatchScore.from_correctness(
                batch.labels, [False] * batch.size
            ),
            model_fingerprint="partial-budget",
            budget=6,
        )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertFalse(result.feedback_ready)

    def test_budget_exhaustion_allows_partial_feedback_when_configured(self):
        plans = make_plans(client_count=3, sample_count=4, batch_size=2)
        result = LUCBSelector(
            SelectionConfig(
                lucb_batch_size=2,
                lucb_top_k=1,
                lucb_budget_fraction=1.0,
                allow_partial_feedback=True,
                weak_class_accuracy=1.0,
            )
        ).select(
            plans,
            lambda _client_id, batch: BatchScore.from_correctness(
                batch.labels, [False] * batch.size
            ),
            model_fingerprint="partial-budget-allowed",
            budget=6,
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertTrue(result.feedback_ready)

    def test_exhausted_arms_are_explicit_and_safe(self):
        plans = make_plans(client_count=3, sample_count=2, batch_size=2)
        flags = {"client-0": False, "client-1": True, "client-2": False}

        def scorer(client_id, batch):
            return BatchScore.from_correctness(batch.labels, [flags[client_id]] * batch.size)

        result = LUCBSelector(
            SelectionConfig(lucb_batch_size=2, lucb_top_k=1, lucb_budget_fraction=1.0)
        ).select(plans, scorer, model_fingerprint="exhausted", budget=6)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stop_reason, "all_batches_exhausted")
        self.assertEqual(result.model_sample_evaluations, 6)
        self.assertEqual(result.selected_client_ids, ("client-0",))

    def test_legacy_full_scan_uses_full_capacity_by_default(self):
        plans = make_plans(client_count=2, sample_count=4, batch_size=2)

        result = LUCBSelector(
            SelectionConfig(
                selection_mode="legacy_full_scan",
                lucb_batch_size=2,
                lucb_top_k=1,
                lucb_budget_fraction=0.25,
            )
        ).select(
            plans,
            lambda client_id, batch: BatchScore.from_correctness(batch.labels, [True] * batch.size),
            model_fingerprint="legacy",
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stop_reason, "legacy_full_scan_complete")
        self.assertEqual(result.model_sample_evaluations, 8)
        self.assertEqual(result.budget, 8)

    def test_strict_reset_discards_prior_model_evidence(self):
        plans = make_plans(client_count=2, sample_count=2, batch_size=2)
        state = SelectionState()
        config = SelectionConfig(lucb_batch_size=2, lucb_top_k=1, lucb_budget_fraction=1.0)
        selector = LUCBSelector(config)

        first = selector.select(
            plans,
            lambda client_id, batch: BatchScore.from_correctness(batch.labels, [False] * batch.size),
            model_fingerprint="first",
            state=state,
            budget=4,
        )
        second = selector.select(
            plans,
            lambda client_id, batch: BatchScore.from_correctness(batch.labels, [True] * batch.size),
            model_fingerprint="second",
            state=state,
            budget=4,
        )

        self.assertEqual(first.model_sample_evaluations, 4)
        self.assertEqual(second.model_sample_evaluations, 4)
        self.assertEqual(state.generation, 2)
        self.assertTrue(all(interval.estimate == 0.0 for interval in second.confidence_bounds.values()))

    def test_decay_priority_is_not_wilson_evidence(self):
        plans = make_plans(client_count=2, sample_count=2, batch_size=2)
        state = SelectionState()
        selector = LUCBSelector(
            SelectionConfig(
                lucb_batch_size=2,
                lucb_top_k=1,
                lucb_budget_fraction=1.0,
                lucb_prioritization="decay_priority",
                priority_decay=0.5,
            )
        )
        selector.select(
            plans,
            lambda client_id, batch: BatchScore.from_correctness(
                batch.labels, [client_id == "client-0"] * batch.size
            ),
            model_fingerprint="before",
            state=state,
            budget=4,
        )
        self.assertTrue(state.priority_by_client)
        result = selector.select(
            plans,
            lambda client_id, batch: BatchScore.from_correctness(batch.labels, [False] * batch.size),
            model_fingerprint="after",
            state=state,
            budget=4,
        )

        self.assertEqual(result.model_sample_evaluations, 4)
        self.assertTrue(all(interval.estimate == 1.0 for interval in result.confidence_bounds.values()))
        self.assertTrue(all(interval.total == 2 for interval in result.confidence_bounds.values()))


class AuditAndProtocolTests(unittest.TestCase):
    def test_audit_sanitizer_removes_raw_content_but_keeps_sample_ids(self):
        record = {
            "sample_ids": ["safe-1"],
            "raw_row": {"feature": 3},
            "nested": {"reference": "private", "token_feedback": ["private"]},
            "selected_clients": ["client-a"],
        }
        cleaned = sanitize_audit_record(record, protocol="abstract")
        encoded = json.dumps(cleaned)
        self.assertIn("sample_ids", cleaned)
        self.assertNotIn("raw_row", encoded)
        self.assertNotIn("reference", encoded)
        self.assertNotIn("token_feedback", encoded)
        self.assertNotIn("raw_row", json.dumps(sanitize_audit_record(record, protocol="legacy")))

        with tempfile.TemporaryDirectory() as directory:
            path = write_audit_json(Path(directory) / "audit.json", record)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), cleaned)

    def test_audit_builder_accepts_serialized_selection_without_raw_data(self):
        selection = {
            "status": "ok",
            "stop_reason": "confidence_separated",
            "model_fingerprint": "fingerprint",
            "model_sample_evaluations": 4,
            "budget": 8,
            "selected_client_ids": ["client-a"],
            "selected_classes": {"client-a": []},
            "confidence_bounds": {},
            "coverage": {},
            "scored_batches": {"client-a": [{"sample_ids": [1, 2], "raw_row": [4, 5]}]},
            "boundary_gap": 0.1,
        }
        record = build_audit_record(
            domain="tabular",
            feedback_protocol="abstract",
            configuration={"seed": 42},
            model_fingerprint="fingerprint",
            selection=selection,
        )
        self.assertEqual(record["domain"], "tabular")
        self.assertEqual(record["scored_batches"]["client-a"][0]["sample_ids"], [1, 2])
        self.assertNotIn("raw_row", json.dumps(record))

    def test_protocol_validation_for_abstract_and_legacy(self):
        payload = {
            "protocol": "abstract",
            "domain": "tabular",
            "weak_labels": [1],
            "feature_ranking": [{"feature": "tenure", "rank": 1, "importance": 0.8}],
        }
        normalized = validate_feedback_payload(payload)
        self.assertEqual(normalized["feature_ranking"][0]["feature"], "tenure")

        with self.assertRaises(FeedbackProtocolError):
            validate_feedback_payload({**payload, "reference": {"row": [1, 2]}})
        with self.assertRaises(FeedbackProtocolError):
            validate_feedback_payload(
                {"protocol": "legacy", "domain": "text", "weak_labels": [1], "tokens": ["x"]}
            )
        legacy = validate_feedback_payload(
            {"protocol": "legacy", "domain": "text", "weak_labels": [1], "tokens": ["x"]},
            allow_raw_feedback=True,
        )
        self.assertEqual(legacy["protocol"], "legacy")

    def test_image_abstract_requires_refine_mode(self):
        payload = {
            "protocol": "abstract",
            "domain": "image",
            "weak_labels": [3],
            "generation_mode": "refine",
            "patch_grid_size": 8,
            "salient_patches": [{"row": 2, "column": 4, "weight": 0.7}],
        }
        self.assertEqual(validate_feedback_payload(payload)["generation_mode"], "refine")
        with self.assertRaises(FeedbackProtocolError):
            validate_feedback_payload({**payload, "generation_mode": "generate"})
        with self.assertRaises(FeedbackProtocolError):
            validate_feedback_payload({**payload, "values": [[0.1]]})
        with self.assertRaises(FeedbackProtocolError):
            validate_feedback_payload({**payload, "client_prob": 0.9})


if __name__ == "__main__":
    unittest.main()
