"""Dependency-light checks for the text track command-line contracts."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import warnings
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVER = _load_module("text_server_entrypoint", "text/server_textdata.py")
CLIENT = _load_module("text_client_entrypoint", "text/client_textdata.py")


class TextEntrypointTests(unittest.TestCase):
    def test_server_defaults_resolve_to_shipped_target_data(self) -> None:
        args = SERVER.validate_startup_args(SERVER.build_parser().parse_args([]))
        self.assertEqual(args.test_csv, ROOT / "text" / "data" / "test_server.csv")
        self.assertEqual(args.selection_mode, "lucb")
        self.assertIsNone(args.lucb_top_k)

    def test_client_derives_stable_id_without_loading_generator(self) -> None:
        args = CLIENT.validate_startup_args(
            CLIENT.build_parser().parse_args(["--generator", "disabled"])
        )
        self.assertEqual(args.client_id, "client0")
        self.assertEqual(args.train_csv, ROOT / "text" / "data" / "client0.csv")

    def test_label_map_fingerprints_match_across_client_and_server(self) -> None:
        mapping = {"news": 1, "sports": 0}
        self.assertEqual(
            SERVER._label_map_fingerprint(mapping),
            CLIENT._label_map_fingerprint(mapping),
        )

    def test_legacy_requires_opt_in_and_warns_when_enabled(self) -> None:
        parser = SERVER.build_parser()
        with self.assertRaises(ValueError):
            SERVER.validate_startup_args(
                parser.parse_args(["--feedback-protocol", "legacy"])
            )
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            args = SERVER.validate_startup_args(
                parser.parse_args(
                    ["--feedback-protocol", "legacy", "--allow-raw-feedback"]
                )
            )
        self.assertEqual(args.feedback_protocol, "legacy")
        self.assertTrue(any("Legacy token feedback" in str(item.message) for item in recorded))

    def test_abstract_prompt_never_needs_tokens(self) -> None:
        prompt = CLIENT.build_abstract_prompt(
            "NEWS",
            [{"position": "beginning", "weight": 1.0}],
        )
        self.assertIn("NEWS", prompt)
        self.assertIn("beginning", prompt)

    def test_server_accepts_only_safe_stable_client_identifiers(self) -> None:
        self.assertEqual(SERVER._stable_client_id("client-01"), "client-01")
        self.assertIsNone(SERVER._stable_client_id("client 01"))
        self.assertIsNone(SERVER._stable_client_id(None))

    def test_global_monitoring_jsonl_is_append_only_and_separate_from_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "global_monitoring.jsonl"
            SERVER._append_jsonl(
                path,
                {
                    "round": 1,
                    "global_monitoring_model_sample_evaluations": 8,
                    "included_in_lucb_budget": False,
                },
            )
            SERVER._append_jsonl(
                path,
                {
                    "round": 2,
                    "global_monitoring_model_sample_evaluations": 8,
                    "included_in_lucb_budget": False,
                },
            )
            entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([entry["round"] for entry in entries], [1, 2])
        self.assertTrue(all(entry["included_in_lucb_budget"] is False for entry in entries))


if __name__ == "__main__":
    unittest.main()
