"""Dependency-only checks for the experimental image entry points.

These tests intentionally do not import Torch, TorchVision, or Flower. They
remain useful in a development environment whose binary packages are broken
while still guarding the pinned runtime contract for the image track.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "image"


def _requirements() -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in (IMAGE_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        entries[name] = version
    return entries


def _string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


class ImageDependencyTests(unittest.TestCase):
    def test_image_runtime_uses_known_compatible_pins(self) -> None:
        self.assertEqual(
            _requirements(),
            {
                "flwr": "1.8.0",
                "protobuf": "4.25.3",
                "torch": "2.2.2",
                "torchvision": "0.17.2",
                "numpy": "1.26.4",
                "tqdm": "4.66.4",
                "Pillow": "10.3.0",
            },
        )

    def test_server_exposes_lucb_and_protocol_configuration(self) -> None:
        literals = _string_literals(IMAGE_DIR / "server.py")
        self.assertTrue(
            {
                "--selection-mode",
                "--lucb-budget-fraction",
                "--lucb-batch-size",
                "--lucb-top-k",
                "--lucb-epsilon",
                "--lucb-delta",
                "--weak-class-accuracy",
                "--lucb-prioritization",
                "--feedback-protocol",
                "--allow-raw-feedback",
                "--validate-config",
            }.issubset(literals)
        )

    def test_server_keeps_annotations_compatible_with_python_38(self) -> None:
        source = (IMAGE_DIR / "server.py").read_text(encoding="utf-8")
        self.assertNotIn("int | None", source)
        self.assertIn("Optional[int]", source)

    def test_compatibility_clients_delegate_to_the_canonical_runtime(self) -> None:
        for client_number in range(2, 10):
            tree = ast.parse(
                (IMAGE_DIR / f"client_{client_number}.py").read_text(encoding="utf-8"),
                filename=f"client_{client_number}.py",
            )
            imports = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "client_1"
            ]
            self.assertTrue(
                any(alias.name == "main" for node in imports for alias in node.names),
                f"client_{client_number}.py must delegate to client_1.main",
            )


if __name__ == "__main__":
    unittest.main()
