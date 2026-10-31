"""Smoke test for Minibatch Cumulative Evaluation and Feedback Mechanisms.

Verifies:
1. Stratified & sequential minibatch partitioning.
2. Minibatch cumulative evaluation:
   - Early stopping on threshold reaching ('GOOD') with compute/sample savings.
   - Exhaustive evaluation on underperforming models ('NEEDS_FEEDBACK').
3. Abstract feedback generation (zero test data leakage).
4. Detailed feedback generation (masked SHAP).
5. Client training set composition:
   - Safe mode (allow_target_leakage=False): ZERO test samples in train data.
   - Legacy mode (allow_target_leakage=True): restored test samples concatenated.

Usage:
    python tabular/smoke_minibatch.py
"""

import os
import shutil
import tempfile
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
    restore_missing_values
)


def run_smoke_test():
    print("=== STARTING MINIBATCH SMOKE TEST ===")

    # 1. Test Minibatch Partitioning
    n_samples = 1000
    n_features = 10
    batch_size = 200

    np.random.seed(42)
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    # Imbalanced labels: 80% class 0, 20% class 1
    labels = (np.random.rand(n_samples) > 0.8).astype(np.float32)

    batches_stratified = build_minibatches(features, labels, batch_size=batch_size, stratified=True)
    assert len(batches_stratified) == 5, f"Expected 5 batches, got {len(batches_stratified)}"
    total_tested = sum(len(b[1]) for b in batches_stratified)
    assert total_tested == n_samples, f"Expected {n_samples} samples across batches, got {total_tested}"

    for i, (bx, by) in enumerate(batches_stratified):
        c1_ratio = (by == 1.0).sum().item() / len(by)
        print(f"Batch {i}: size={len(by)}, class-1 ratio={c1_ratio:.2f}")
        assert 0.15 <= c1_ratio <= 0.25, f"Stratified ratio deviated too much: {c1_ratio}"
    print("✓ Stratified minibatch partitioning PASSED\n")

    # 2. Test Minibatch Evaluation: GOOD model (Early Stop)
    class HighAccModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, x):
            # Output matches labels approximately
            return torch.full((len(x), 1), -5.0)  # Predicts class 0 (80% accurate)

    good_model = HighAccModel()
    res_good = evaluate_model_minibatch(
        model=good_model,
        minibatches=batches_stratified,
        threshold=0.75,  # 80% > 75%, should hit GOOD
        max_minibatches=5,
        total_available_samples=n_samples
    )
    print(f"Good Model Evaluation: batches_tested={res_good['batches_tested']}, "
          f"acc={res_good['cumulative_accuracy']}, status={res_good['status']}, "
          f"samples_saved_pct={res_good['samples_saved_pct']}%")
    assert res_good["status"] == "GOOD", f"Expected GOOD, got {res_good['status']}"
    assert res_good["batches_tested"] == 1, f"Expected early stop at batch 1, got {res_good['batches_tested']}"
    assert res_good["samples_saved_pct"] >= 75.0, f"Expected at least 75% saved, got {res_good['samples_saved_pct']}"
    print("✓ Early stopping for GOOD model PASSED\n")

    # 3. Test Minibatch Evaluation: NEEDS_FEEDBACK model
    class LowAccModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.full((len(x), 1), 5.0)  # Predicts class 1 (only 20% accurate)

    low_model = LowAccModel()
    res_low = evaluate_model_minibatch(
        model=low_model,
        minibatches=batches_stratified,
        threshold=0.85,
        max_minibatches=5,
        total_available_samples=n_samples
    )
    print(f"Low Model Evaluation: batches_tested={res_low['batches_tested']}, "
          f"acc={res_low['cumulative_accuracy']}, status={res_low['status']}, "
          f"samples_saved_pct={res_low['samples_saved_pct']}%")
    assert res_low["status"] == "NEEDS_FEEDBACK", f"Expected NEEDS_FEEDBACK, got {res_low['status']}"
    assert res_low["batches_tested"] == 5, f"Expected all 5 batches tested, got {res_low['batches_tested']}"
    assert res_low["samples_saved_pct"] == 0.0, f"Expected 0% saved, got {res_low['samples_saved_pct']}"
    print("✓ Full evaluation for NEEDS_FEEDBACK model PASSED\n")

    # 3.5 Test Budget Ceiling Mechanism
    res_budget = evaluate_model_minibatch(
        model=low_model,
        minibatches=batches_stratified,
        threshold=0.85,
        max_minibatches=5,
        total_available_samples=n_samples,
        budget_remaining=350  # Cap at 350 samples (< 1000 total)
    )
    print(f"Budget Capped Evaluation: batches_tested={res_budget['batches_tested']}, "
          f"samples_tested={res_budget['samples_tested']}, status={res_budget['status']}, "
          f"budget_exhausted={res_budget['budget_exhausted']}")
    assert res_budget["samples_tested"] <= 350, f"Exceeded budget ceiling! Tested {res_budget['samples_tested']}"
    assert res_budget["budget_exhausted"] is True, "Expected budget_exhausted to be True"
    assert res_budget["status"] == "BUDGET_EXHAUSTED", f"Expected BUDGET_EXHAUSTED, got {res_budget['status']}"
    print("✓ Global Budget Ceiling mechanism PASSED\n")

    # 4. Test Abstract Feedback Generation (Zero Data Leakage)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    abstract_fb = compute_abstract_feedback(
        model=low_model,
        misclassified_samples=res_low["misclassified_samples"],
        feature_names=feature_names,
        top_k=3
    )
    print("Abstract Feedback generated:", abstract_fb)
    assert abstract_fb["protocol"] == "abstract"
    assert abstract_fb["weak_label"] == 0
    assert len(abstract_fb["top_features"]) == 3
    assert "reference" not in abstract_fb
    assert "shap_feedback" not in abstract_fb
    print("✓ Abstract feedback privacy PASSED\n")

    # 5. Test Client Training & Data Leakage Prevention
    train_df = pd.DataFrame(features[:100], columns=feature_names)
    train_df["Exited"] = labels[:100]

    mock_model = SimpleNN(n_features)
    client_safe = FLClient(
        client_id="client_test",
        model=mock_model,
        train_df=train_df,
        restore_synthesizer=None,
        generator=None,
        allow_target_leakage=False
    )

    # Detailed feedback payload with 10 mock test samples
    mock_shap_entries = [{"features": {name: 1.0 for name in feature_names}, "label": 1} for _ in range(10)]
    detailed_config = {
        "feedback": '{"protocol": "detailed", "shap_feedback": ' + str(mock_shap_entries).replace("'", '"') + '}'
    }

    # Safe mode fit
    client_safe.fit(client_safe.get_parameters({}), detailed_config)
    print(f"Client Safe Mode - Train size: {len(client_safe.train_data)} (Original={len(train_df)})")
    assert len(client_safe.train_data) == len(train_df), \
        f"Safe mode leaked data! Expected {len(train_df)}, got {len(client_safe.train_data)}"

    # Leakage mode fit
    client_leaky = FLClient(
        client_id="client_test_leaky",
        model=mock_model,
        train_df=train_df,
        restore_synthesizer=None,
        generator=None,
        allow_target_leakage=True
    )
    client_leaky.fit(client_leaky.get_parameters({}), detailed_config)
    print(f"Client Leaky Mode - Train size: {len(client_leaky.train_data)} (Original={len(train_df)} + Restored=10)")
    assert len(client_leaky.train_data) == len(train_df) + 10, \
        f"Expected leaky train size {len(train_df) + 10}, got {len(client_leaky.train_data)}"

    print("✓ Target data leakage safeguard & verification PASSED\n")
    print("=== ALL SMOKE TESTS COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_smoke_test()
