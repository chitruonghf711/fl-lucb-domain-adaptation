"""Unit tests for Minibatch Cumulative Evaluation and Data Leakage Safeguards."""

import json
import unittest
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tabular.server_tabular import (
    SimpleNN,
    build_minibatches,
    evaluate_model_minibatch,
    compute_abstract_feedback,
    compute_detailed_feedback,
)
from tabular.client_tabular import (
    FLClient,
    feedback_to_dataframe,
    restore_missing_values,
)


class MinibatchPartitionTests(unittest.TestCase):
    def test_build_minibatches_stratified(self):
        n_samples = 600
        n_features = 5
        batch_size = 150

        np.random.seed(123)
        features = np.random.randn(n_samples, n_features).astype(np.float32)
        # Class 0: 450 (75%), Class 1: 150 (25%)
        labels = np.array([0.0] * 450 + [1.0] * 150, dtype=np.float32)

        batches = build_minibatches(features, labels, batch_size=batch_size, stratified=True)
        self.assertEqual(len(batches), 4)

        total_tested = sum(len(b[1]) for b in batches)
        self.assertEqual(total_tested, n_samples)

        for b_x, b_y in batches:
            self.assertIn(len(b_y), [149, 150, 151])
            c1_count = (b_y == 1.0).sum().item()
            # 150 samples * 25% = ~37-38
            self.assertIn(c1_count, [37, 38])

    def test_build_minibatches_sequential(self):
        features = np.arange(20).reshape(10, 2).astype(np.float32)
        labels = np.zeros(10, dtype=np.float32)
        batches = build_minibatches(features, labels, batch_size=3, stratified=False)
        self.assertEqual(len(batches), 4)
        self.assertEqual([len(b[1]) for b in batches], [3, 3, 3, 1])


class MinibatchEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.n_features = 4
        self.n_samples = 500
        self.features = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        # 80% class 0, 20% class 1
        self.labels = np.array([0.0] * 400 + [1.0] * 100, dtype=np.float32)
        self.batches = build_minibatches(self.features, self.labels, batch_size=100, stratified=True)

    def test_early_stopping_good_model(self):
        # Model that predicts class 0 for everything -> 80% accuracy on every batch
        class PredictZeroModel(nn.Module):
            def forward(self, x):
                return torch.full((len(x), 1), -10.0)

        model = PredictZeroModel()
        res = evaluate_model_minibatch(
            model=model,
            minibatches=self.batches,
            threshold=0.75,  # 80% >= 75%
            max_minibatches=5,
            total_available_samples=self.n_samples
        )

        self.assertEqual(res["status"], "GOOD")
        self.assertEqual(res["batches_tested"], 1)
        self.assertEqual(res["samples_tested"], 100)
        self.assertEqual(res["cumulative_accuracy"], 0.8)
        self.assertEqual(res["samples_saved_pct"], 80.0)

    def test_exhaustive_evaluation_underperforming_model(self):
        # Model that predicts class 1 for everything -> only 20% accuracy
        class PredictOneModel(nn.Module):
            def forward(self, x):
                return torch.full((len(x), 1), 10.0)

        model = PredictOneModel()
        res = evaluate_model_minibatch(
            model=model,
            minibatches=self.batches,
            threshold=0.85,
            max_minibatches=5,
            total_available_samples=self.n_samples
        )

        self.assertEqual(res["status"], "NEEDS_FEEDBACK")
        self.assertEqual(res["batches_tested"], 5)
        self.assertEqual(res["samples_tested"], 500)
        self.assertEqual(res["cumulative_accuracy"], 0.2)
        self.assertEqual(res["samples_saved_pct"], 0.0)
        self.assertEqual(len(res["misclassified_samples"]), 400)

    def test_evaluation_stops_at_budget_ceiling(self):
        class PredictOneModel(nn.Module):
            def forward(self, x):
                return torch.full((len(x), 1), 10.0)

        model = PredictOneModel()
        res = evaluate_model_minibatch(
            model=model,
            minibatches=self.batches,
            threshold=0.85,
            max_minibatches=5,
            total_available_samples=self.n_samples,
            budget_remaining=250  # Only allow 250 samples
        )

        self.assertEqual(res["status"], "BUDGET_EXHAUSTED")
        self.assertLessEqual(res["samples_tested"], 250)
        self.assertTrue(res["budget_exhausted"])
        self.assertLess(res["batches_tested"], 5)


class FeedbackAndDataLeakageTests(unittest.TestCase):
    def test_abstract_feedback_does_not_contain_raw_samples(self):
        misclassified = [
            (torch.tensor([1.0, 2.0, 3.0]), 1, 0),
            (torch.tensor([1.5, 2.5, 3.5]), 1, 0),
            (torch.tensor([0.0, 0.0, 0.0]), 0, 1),
        ]
        feature_names = ["f1", "f2", "f3"]
        model = SimpleNN(3)

        abstract_fb = compute_abstract_feedback(
            model=model,
            misclassified_samples=misclassified,
            feature_names=feature_names,
            top_k=2
        )

        self.assertEqual(abstract_fb["protocol"], "abstract")
        self.assertEqual(abstract_fb["weak_label"], 1)
        self.assertEqual(len(abstract_fb["top_features"]), 2)
        self.assertNotIn("reference", abstract_fb)
        self.assertNotIn("shap_feedback", abstract_fb)

    def test_detailed_feedback_structure(self):
        misclassified = [
            (torch.tensor([1.0, 2.0, 3.0]), 1, 0),
            (torch.tensor([1.5, 2.5, 3.5]), 1, 0),
        ]
        feature_names = ["f1", "f2", "f3"]
        model = SimpleNN(3)

        detailed_fb = compute_detailed_feedback(
            model=model,
            misclassified_samples=misclassified,
            feature_names=feature_names,
            target_column="Exited",
            top_k=2,
            limit=10
        )

        self.assertEqual(detailed_fb["protocol"], "detailed")
        self.assertEqual(detailed_fb["weak_label"], 1)
        self.assertEqual(len(detailed_fb["shap_feedback"]), 2)
        self.assertEqual(len(detailed_fb["reference"]), 2)

    def test_client_training_prevents_target_leakage_by_default(self):
        n_features = 3
        feature_names = ["f1", "f2", "f3"]
        train_df = pd.DataFrame({
            "f1": [0.1, 0.2, 0.3],
            "f2": [0.4, 0.5, 0.6],
            "f3": [0.7, 0.8, 0.9],
            "Exited": [0, 1, 0]
        })
        model = SimpleNN(n_features)

        client_safe = FLClient(
            client_id="c_safe",
            model=model,
            train_df=train_df,
            restore_synthesizer=None,
            generator=None,
            allow_target_leakage=False
        )

        mock_shap = [{"features": {"f1": 1.0, "f2": 2.0, "f3": 3.0}, "label": 1}] * 5
        config = {
            "feedback": json.dumps({"protocol": "detailed", "shap_feedback": mock_shap})
        }

        client_safe.fit(client_safe.get_parameters({}), config)
        # Safe mode must NOT add restored samples to train data
        self.assertEqual(len(client_safe.train_data), len(train_df))

        # Leaky mode
        client_leaky = FLClient(
            client_id="c_leaky",
            model=model,
            train_df=train_df,
            restore_synthesizer=None,
            generator=None,
            allow_target_leakage=True
        )
        client_leaky.fit(client_leaky.get_parameters({}), config)
        self.assertEqual(len(client_leaky.train_data), len(train_df) + 5)

    def test_client_time_summary_tracking(self):
        import tempfile
        import os

        n_features = 3
        train_df = pd.DataFrame({
            "f1": [0.1, 0.2, 0.3],
            "f2": [0.4, 0.5, 0.6],
            "f3": [0.7, 0.8, 0.9],
            "Exited": [0, 1, 0]
        })
        model = SimpleNN(n_features)

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = FLClient(
                client_id="client_timing_test",
                model=model,
                train_df=train_df,
                restore_synthesizer=None,
                generator=None,
                client_log_dir=tmp_dir
            )
            config = {"server_round": 1}
            client.fit(client.get_parameters({}), config)

            self.assertEqual(len(client.time_logs), 1)
            log = client.time_logs[0]
            self.assertEqual(log["round"], 1)
            self.assertIn("total_fit_time_sec", log)
            self.assertIn("train_time_sec", log)
            self.assertIn("gen_time_sec", log)
            self.assertIn("restore_time_sec", log)
            self.assertEqual(log["train_samples"], 3)

            csv_file = os.path.join(tmp_dir, "client_time_summary.csv")
            self.assertTrue(os.path.exists(csv_file))
            saved_df = pd.read_csv(csv_file)
            self.assertEqual(len(saved_df), 1)
            self.assertEqual(saved_df["round"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
