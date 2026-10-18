"""Image feedback audit tests without Flower, TorchVision, or external assets."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]


def load_image_server() -> types.ModuleType:
    """Load the server with minimal Flower and TorchVision interfaces."""

    flwr = types.ModuleType("flwr")
    flwr_common = types.ModuleType("flwr.common")
    flwr_server = types.ModuleType("flwr.server")
    flwr_strategy = types.ModuleType("flwr.server.strategy")

    class FedAvg:
        def __init__(self, **kwargs):
            del kwargs

    flwr_common.FitIns = lambda parameters, config: (parameters, config)
    flwr_common.Parameters = object
    flwr_common.ndarrays_to_parameters = lambda values: values
    flwr_common.parameters_to_ndarrays = lambda values: values
    flwr_strategy.FedAvg = FedAvg
    flwr_server.strategy = flwr_strategy
    flwr.server = flwr_server
    flwr.common = flwr_common

    torchvision = types.ModuleType("torchvision")
    torchvision_datasets = types.ModuleType("torchvision.datasets")

    class ImageFolder(Dataset):
        pass

    torchvision_datasets.ImageFolder = ImageFolder
    torchvision.transforms = types.SimpleNamespace()

    replacements = {
        "flwr": flwr,
        "flwr.common": flwr_common,
        "flwr.server": flwr_server,
        "flwr.server.strategy": flwr_strategy,
        "torchvision": torchvision,
        "torchvision.datasets": torchvision_datasets,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    module_name = "image_server_lucb_test"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "image" / "server.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    return module


class TinyImageDataset(Dataset):
    """Minimal target set used only to construct an audit-capable strategy."""

    def __init__(self) -> None:
        self.samples = [
            (torch.zeros(3, 32, 32), 0),
            (torch.ones(3, 32, 32), 1),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def partial_selection() -> types.SimpleNamespace:
    """Create a selection object that must never release feedback."""

    return types.SimpleNamespace(
        status="partial",
        stop_reason="budget_exhausted",
        feedback_ready=False,
        audit={"configuration": {"confidence_allocation": {"method": "test"}}},
        model_fingerprint="partial-model",
        model_sample_evaluations=8,
        budget=8,
        selected_client_ids=("client-a",),
        selected_classes={"client-a": ()},
        confidence_bounds={},
        coverage={"client-a": {"scored_sample_count": 8}},
        scored_batches={},
        boundary_gap=-0.1,
    )


class ImageAuditIntegrationTests(unittest.TestCase):
    """Verify delivery gating and standardized sanitized audit records."""

    def setUp(self) -> None:
        self.server = load_image_server()
        self.strategy = self.server.FeedbackFedAvg(
            test_set=TinyImageDataset(),
            target_layer_of=lambda model: model,
            device=torch.device("cpu"),
            feedback_protocol="abstract",
        )

    def test_partial_selection_cannot_build_feedback_and_audit_records_reason(self) -> None:
        selection = partial_selection()
        feedback, explanation_evaluations = self.strategy._build_lucb_feedback({}, selection)
        self.assertEqual(feedback, {})
        self.assertEqual(explanation_evaluations, 0)
        delivery = self.strategy._lucb_delivery_state(selection)
        self.assertEqual(delivery["status"], "not_delivered")
        self.assertEqual(delivery["reason"], "selection_status_not_ok")

        with tempfile.TemporaryDirectory(prefix="image_lucb_audit_") as directory:
            previous_log_dir = self.server.SERVER_LOG_DIR
            self.server.SERVER_LOG_DIR = directory
            try:
                self.strategy._write_lucb_audit(3, selection, 0, delivery)
                audit = json.loads(
                    (Path(directory) / "lucb_audit_round_3.json").read_text(encoding="utf-8")
                )
            finally:
                self.server.SERVER_LOG_DIR = previous_log_dir

        self.assertEqual(audit["metadata"]["feedback_delivery"]["status"], "not_delivered")
        self.assertEqual(audit["metadata"]["feedback_delivery"]["reason"], "selection_status_not_ok")
        self.assertEqual(audit["explanation_cost"]["grad_cam_evaluations"], 0)

    def test_legacy_full_scan_audit_is_standardized_and_sanitized(self) -> None:
        model = torch.nn.Linear(1, 1)
        raw_feedback = {
            "client-a": [{"sample_id": 7, "values": [[[0.1]]], "label": 1}],
        }
        with tempfile.TemporaryDirectory(prefix="image_legacy_audit_") as directory:
            previous_log_dir = self.server.SERVER_LOG_DIR
            self.server.SERVER_LOG_DIR = directory
            try:
                self.strategy._write_legacy_full_scan_audit(
                    4, {"client-a": model}, ["client-a"], raw_feedback, "forced test"
                )
                text = (Path(directory) / "legacy_full_scan_audit_round_4.json").read_text(
                    encoding="utf-8"
                )
            finally:
                self.server.SERVER_LOG_DIR = previous_log_dir

        audit = json.loads(text)
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["stop_reason"], "legacy_full_scan_complete")
        self.assertEqual(audit["explanation_cost"]["grad_cam_evaluations"], 1)
        self.assertNotIn('"values"', text)
        self.assertNotIn('"sample_id"', text)

    def test_lucb_does_not_compare_train_accuracy_to_stale_local_accuracy(self) -> None:
        result = types.SimpleNamespace(metrics={"train_acc": 0.95})
        proxy = types.SimpleNamespace(cid="client-a")
        self.strategy.local_acc = {"client-a": 0.10}
        self.strategy.selection_mode = "lucb"
        trigger, reason = self.strategy._should_trigger(10, [(proxy, result)])
        self.assertFalse(trigger)
        self.assertEqual(reason, "not needed")

        self.strategy.selection_mode = "legacy_full_scan"
        trigger, reason = self.strategy._should_trigger(10, [(proxy, result)])
        self.assertTrue(trigger)
        self.assertTrue(reason.startswith("overfit:"))


if __name__ == "__main__":
    unittest.main()
