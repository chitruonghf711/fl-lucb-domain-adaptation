"""Server-side tabular integration tests without Flower or SHAP runtime dependencies."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def load_server_module() -> types.ModuleType:
    """Load the integration module with minimal interface-only Flower and SHAP stubs."""

    flwr = types.ModuleType("flwr")

    class FedAvg:
        def __init__(self, **kwargs):
            del kwargs

    flwr.server = types.SimpleNamespace(strategy=types.SimpleNamespace(FedAvg=FedAvg))
    flwr.common = types.SimpleNamespace()
    shap = types.ModuleType("shap")
    module_name = "tabular_server_lucb_test"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tabular" / "server_tabular.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_flwr = sys.modules.get("flwr")
    previous_shap = sys.modules.get("shap")
    sys.modules["flwr"] = flwr
    sys.modules["shap"] = shap
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_flwr is None:
            sys.modules.pop("flwr", None)
        else:
            sys.modules["flwr"] = previous_flwr
        if previous_shap is None:
            sys.modules.pop("shap", None)
        else:
            sys.modules["shap"] = previous_shap
    return module


class TabularLUCBIntegrationTests(unittest.TestCase):
    """Verify the server callback only emits safe, budgeted abstract feedback."""

    def test_lucb_callback_is_budgeted_and_abstract(self) -> None:
        server = load_server_module()
        with tempfile.TemporaryDirectory(prefix="tabular_lucb_test_") as directory:
            server.CONFIG.update({
                "selection_mode": "lucb",
                "lucb_budget_fraction": 0.25,
                "lucb_batch_size": 64,
                "lucb_top_k": 3,
                "lucb_epsilon": 1.0,
                "lucb_delta": 0.05,
                "weak_class_accuracy": 0.85,
                "seed": 42,
                "lucb_prioritization": "strict",
                "feedback_protocol": "abstract",
                "allow_raw_feedback": False,
                "max_feedback_per_client": 10,
                "top_k_features": 2,
                "log_dir": directory,
            })
            server.refresh_run_paths(server.CONFIG)
            for key in ("server_log_dir", "server_feedback_dir", "server_audit_dir"):
                os.makedirs(server.CONFIG[key], exist_ok=True)
            server.build_abstract_tabular_payload = lambda client_id, model, errors, names: {
                "protocol": "abstract",
                "domain": "tabular",
                "client_id": client_id,
                "weak_labels": sorted(errors),
                "feature_ranking": [{"feature": "0", "rank": 1, "importance": 0.5}],
            }
            first_feature = torch.linspace(-1.0, 1.0, 256)
            inputs = torch.stack([first_feature, torch.zeros_like(first_feature)], dim=1)
            targets = (first_feature > 0).to(torch.float32)
            strategy = server.FeedbackFedAvg(
                server.SimpleNN(2),
                DataLoader(TensorDataset(inputs, targets), batch_size=32, shuffle=False),
                ["0", "1"],
                2,
            )
            models = {}
            specifications = {
                "client_1": (10.0, 0.0),
                "client_2": (10.0, 3.0),
                "client_3": (10.0, 6.0),
                "client_4": (-10.0, 0.0),
                "client_5": (10.0, -6.0),
            }
            for client_id, (weight, bias) in specifications.items():
                model = torch.nn.Linear(2, 1)
                model.weight.data[:] = torch.tensor([[weight, 0.0]])
                model.bias.data.fill_(bias)
                models[client_id] = model

            _, summary = strategy._run_lucb_feedback(models)

            self.assertLessEqual(summary["model_sample_evaluations"], summary["budget"])
            self.assertIn("eval_time_sec", summary)
            self.assertIn("samples_saved_pct", summary)
            self.assertIn("samples_tested", summary)
            self.assertTrue(strategy.feedback_storage)
            self.assertTrue(all("reference" not in bundle for bundle in strategy.feedback_storage.values()))
            audit = (Path(server.CONFIG["server_audit_dir"]) / "feedback_round_0.json").read_text(encoding="utf-8")
            self.assertNotIn('"reference"', audit)
            self.assertNotIn('"raw"', audit)

    def test_client_time_summary_tracking(self) -> None:
        import pandas as pd
        from tabular.client_tabular import FLClient, SimpleNN

        with tempfile.TemporaryDirectory(prefix="client_timing_") as tmp_dir:
            model = SimpleNN(2)
            train_df = pd.DataFrame({
                "f1": [0.1, 0.2, 0.3],
                "f2": [0.4, 0.5, 0.6],
                "Exited": [0, 1, 0]
            })
            client = FLClient(
                client_id="client_test",
                model=model,
                train_df=train_df,
                restore_synthesizer=None,
                generator=None,
                client_log_dir=tmp_dir
            )
            # Run in smoke_no_generation mode
            import tabular.client_tabular as ctab
            orig_smoke = ctab.CONFIG["smoke_no_generation"]
            ctab.CONFIG["smoke_no_generation"] = True
            try:
                client.fit(client.get_parameters({}), {"server_round": 1})
            finally:
                ctab.CONFIG["smoke_no_generation"] = orig_smoke

            self.assertEqual(len(client.time_logs), 1)
            entry = client.time_logs[0]
            self.assertEqual(entry["round"], 1)
            self.assertIn("total_fit_time_sec", entry)
            self.assertIn("train_time_sec", entry)
            self.assertIn("gen_time_sec", entry)
            self.assertIn("restore_time_sec", entry)
            self.assertEqual(entry["train_samples"], 3)

            csv_file = os.path.join(tmp_dir, "client_time_summary.csv")
            self.assertTrue(os.path.exists(csv_file))
            df = pd.read_csv(csv_file)
            self.assertEqual(len(df), 1)
            self.assertEqual(df["round"].iloc[0], 1)

    def test_duplicate_stable_id_guard_clears_pending_feedback(self) -> None:
        server = load_server_module()
        with tempfile.TemporaryDirectory(prefix="tabular_routing_guard_") as directory:
            server.CONFIG.update({"log_dir": directory, "feedback_protocol": "abstract"})
            server.refresh_run_paths(server.CONFIG)
            for key in ("server_log_dir", "server_feedback_dir", "server_audit_dir"):
                os.makedirs(server.CONFIG[key], exist_ok=True)
            strategy = server.FeedbackFedAvg(
                server.SimpleNN(2),
                DataLoader(
                    TensorDataset(torch.zeros(4, 2), torch.tensor([0.0, 1.0, 0.0, 1.0])),
                    batch_size=2,
                ),
                ["0", "1"],
                2,
            )
            strategy.feedback_storage = {"duplicate": {"feedback": {"protocol": "abstract"}}}
            with self.assertWarns(RuntimeWarning):
                strategy._record_duplicate_client_ids(["duplicate"])
            self.assertEqual(strategy.feedback_storage, {})
            self.assertEqual(strategy._last_selection_summary["stop_reason"], "duplicate_stable_client_id")


if __name__ == "__main__":
    unittest.main()
