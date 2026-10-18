"""Federated text client with privacy-preserving feedback-driven generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def _nonnegative_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Value must be in the interval [0, 1].")
    return parsed


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one stable text FL client with abstract or opt-in legacy feedback."
    )
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--train-csv", default="data/client0.csv")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--run-dir", default="runs/text_clients")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Defaults to <run-dir>/<client-id>/checkpoints.",
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="Use the label map emitted by the server for a stable global class order.",
    )
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=_positive_int, default=64)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--epochs", type=_positive_int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, or a valid torch device string.",
    )
    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument("--balance-data", dest="balance_data", action="store_true")
    balance_group.add_argument("--no-balance-data", dest="balance_data", action="store_false")
    parser.set_defaults(balance_data=True)

    parser.add_argument(
        "--feedback-protocol", choices=("abstract", "legacy"), default="abstract"
    )
    parser.add_argument(
        "--allow-raw-feedback",
        action="store_true",
        help="Permit legacy token feedback. Abstract feedback never needs this flag.",
    )
    parser.add_argument(
        "--generator",
        choices=("gemma", "disabled"),
        default="gemma",
        help="Use disabled for a pure FL run with no synthetic generation.",
    )
    parser.add_argument("--generator-model", default="google/gemma-2b-it")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--generation-batch-size", type=_positive_int, default=4)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=60)
    parser.add_argument(
        "--generated-per-weak-label", type=_positive_int, default=1
    )
    parser.add_argument("--local-gen-ratio", type=_nonnegative_fraction, default=0.0)
    parser.add_argument(
        "--save-merged-data",
        action="store_true",
        help="Save the local real-plus-synthetic data snapshot after generation.",
    )
    return parser


def _validate_csv_header(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = {"text", "label"}.difference(header)
    if missing:
        raise ValueError(
            f"Training CSV must contain text and label columns; missing: {sorted(missing)}."
        )


def _derive_client_id(train_csv: Path) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "-", train_csv.stem).strip("-.")
    return candidate or "text-client"


def validate_startup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve user input without loading models or contacting external services."""
    args.train_csv = _resolve_path(args.train_csv)
    args.run_dir = _resolve_path(args.run_dir)
    args.label_map = _resolve_path(args.label_map) if args.label_map else None
    args.client_id = args.client_id or _derive_client_id(args.train_csv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.client_id):
        raise ValueError(
            "--client-id may only contain letters, digits, periods, underscores, and hyphens."
        )
    args.client_dir = args.run_dir / args.client_id
    args.checkpoint_dir = (
        _resolve_path(args.checkpoint_dir)
        if args.checkpoint_dir
        else args.client_dir / "checkpoints"
    )

    if not args.train_csv.is_file():
        raise ValueError(f"Training CSV does not exist: {args.train_csv}")
    _validate_csv_header(args.train_csv)
    if args.feedback_protocol == "legacy" and not args.allow_raw_feedback:
        raise ValueError("The legacy feedback protocol requires --allow-raw-feedback.")
    if args.feedback_protocol == "legacy":
        warnings.warn(
            "Legacy token feedback is enabled. Runtime payloads may contain token strings; "
            "client audit receipts do not persist them.",
            RuntimeWarning,
        )
    if args.allow_raw_feedback and args.feedback_protocol != "legacy":
        raise ValueError(
            "--allow-raw-feedback is only valid with --feedback-protocol legacy."
        )
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be greater than zero.")
    if args.generator == "disabled" and args.local_gen_ratio > 0.0:
        raise ValueError("--local-gen-ratio requires --generator gemma.")
    if args.generator == "gemma" and not os.environ.get(args.hf_token_env):
        raise ValueError(
            f"Generator credentials are missing. Set the {args.hf_token_env} environment variable "
            "or pass --generator disabled."
        )
    return args


def _load_label_map(labels: Sequence[object], label_map_path: Path | None) -> dict[str, int]:
    local_labels = [str(label) for label in labels]
    if label_map_path is None:
        mapping = {label: index for index, label in enumerate(sorted(set(local_labels)))}
        print(
            "[Client] Warning: no --label-map was supplied; using a deterministic local map. "
            "Pass the server label map for multi-client runs."
        )
        return mapping
    if not label_map_path.is_file():
        raise ValueError(f"Label map does not exist: {label_map_path}")
    with label_map_path.open(encoding="utf-8") as handle:
        raw_mapping = json.load(handle)
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Label map must be a JSON object: {label_map_path}")
    mapping = {str(key): int(value) for key, value in raw_mapping.items()}
    missing = sorted(set(local_labels).difference(mapping))
    if missing:
        raise ValueError(
            f"Label map {label_map_path} does not define local labels: {missing[:10]}"
        )
    return mapping


def _label_map_fingerprint(mapping: Mapping[str, int]) -> str:
    """Return the canonical label-map identity expected by the server."""

    canonical = json.dumps(
        {str(key): int(value) for key, value in mapping.items()},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _select_device(torch_module: Any, raw_device: str) -> Any:
    if raw_device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    device = torch_module.device(raw_device)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise ValueError("A CUDA device was requested but CUDA is not available.")
    return device


def _clean_token(token: str) -> str:
    return token.lstrip("#").replace("▁", "").strip("ĠĊ")


def build_abstract_prompt(label_name: str, salient_positions: Sequence[Mapping[str, Any]]) -> str:
    """Prompt from labels and coarse positions only; no server text is required."""
    positions = [str(item["position"]) for item in salient_positions if item.get("weight", 0.0) > 0.0]
    location_text = ", ".join(sorted(set(positions))) or "the headline"
    return (
        f"Write one realistic news headline of around thirty words for the category '{label_name}'. "
        f"Place its most specific details around {location_text}. Natural language only:"
    )


def build_legacy_prompt(label_name: str, tokens: Sequence[str]) -> str:
    cleaned = [f"'{_clean_token(token)}'" for token in tokens if _clean_token(token)]
    token_text = ", ".join(cleaned)
    return (
        f"Write one realistic news headline of around thirty words for the category '{label_name}' "
        f"that naturally uses these concepts: {token_text}. Natural language only:"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temporary_path.replace(path)


def run_client(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import flwr as fl
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DistilBertForSequenceClassification,
            pipeline,
        )
        from fl_lucb import FeedbackProtocolError, validate_feedback_payload
    except Exception as exc:
        raise RuntimeError(
            "Could not load the text runtime dependencies. Install the pinned text/requirements.txt "
            "in a compatible environment and ensure the repository root is importable."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.client_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(torch, args.device)

    frame = pd.read_csv(args.train_csv)
    frame = frame.dropna(subset=["text", "label"]).reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("Training CSV must contain at least two usable rows.")
    label_map = _load_label_map(frame["label"].tolist(), args.label_map)
    label_map_fingerprint = _label_map_fingerprint(label_map)
    label_names = {value: key for key, value in label_map.items()}
    base_texts = frame["text"].astype(str).tolist()
    base_labels = [label_map[str(item)] for item in frame["label"].tolist()]
    num_labels = max(label_map.values()) + 1
    if num_labels < 2:
        raise ValueError("At least two labels are required for sequence classification.")

    classifier_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    class DistilBertClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = DistilBertForSequenceClassification.from_pretrained(
                args.model_name, num_labels=num_labels
            )

        def forward(self, input_ids: Any, attention_mask: Any = None, labels: Any = None) -> Any:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

    class HeadlineGenerator:
        """Load the gated generator lazily so no model is downloaded without feedback."""

        def __init__(self) -> None:
            self._generator: Any = None

        def _load(self) -> Any:
            if self._generator is not None:
                return self._generator
            token = os.environ.get(args.hf_token_env)
            if not token:
                raise RuntimeError(
                    f"Generator credentials disappeared from environment variable {args.hf_token_env}."
                )
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.generator_model, token=token)
                model_dtype = torch.float16 if device.type == "cuda" else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    args.generator_model,
                    token=token,
                    torch_dtype=model_dtype,
                    device_map="auto" if device.type == "cuda" else None,
                    low_cpu_mem_usage=True,
                )
                if device.type != "cuda":
                    model.to(device)
                self._generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    top_p=0.9,
                    return_full_text=False,
                    batch_size=args.generation_batch_size,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not load the configured generator. Verify Hugging Face access, model assets, "
                    "network availability, and GPU memory; or use --generator disabled."
                ) from exc
            return self._generator

        def generate(self, prompts: Sequence[str]) -> list[str]:
            if not prompts:
                return []
            output = self._load()(list(prompts), temperature=0.8, do_sample=True)
            generated: list[str] = []
            for item in output:
                candidate = item[0] if isinstance(item, list) else item
                text = str(candidate.get("generated_text", "")).strip()
                if len(text.split()) >= 8:
                    generated.append(" ".join(text.split()[:40]))
            return generated

    def get_parameters(model: Any) -> list[Any]:
        return [item.detach().cpu().numpy() for item in model.state_dict().values()]

    def set_parameters(model: Any, parameters: Sequence[Any]) -> None:
        state_dict = model.state_dict()
        if len(state_dict) != len(parameters):
            raise ValueError(
                f"Parameter length mismatch: model expects {len(state_dict)}, received {len(parameters)}."
            )
        for (key, existing), parameter in zip(state_dict.items(), parameters):
            state_dict[key] = torch.tensor(parameter, dtype=existing.dtype)
        model.load_state_dict(state_dict, strict=True)

    model = DistilBertClassifier().to(device)

    def checkpoint_path(round_index: int) -> Path:
        return args.checkpoint_dir / f"round_{round_index}.pt"

    def load_latest_checkpoint() -> tuple[int, dict[str, Any]]:
        checkpoint_files = sorted(
            args.checkpoint_dir.glob("round_*.pt"),
            key=lambda item: int(item.stem.split("_")[1]),
        )
        if not checkpoint_files:
            return 0, {}
        latest = checkpoint_files[-1]
        checkpoint = torch.load(latest, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"[Client {args.client_id}] Loaded checkpoint {latest}")
        return int(checkpoint["round"]), dict(checkpoint.get("meta", {}))

    def save_checkpoint(round_index: int, metadata: Mapping[str, Any]) -> None:
        path = checkpoint_path(round_index)
        torch.save(
            {"round": round_index, "model": model.state_dict(), "meta": dict(metadata)},
            path,
        )
        print(f"[Client {args.client_id}] Saved checkpoint {path}")

    last_round, _checkpoint_metadata = load_latest_checkpoint()
    texts = list(base_texts)
    labels = list(base_labels)

    def load_past_synthetic(until_round: int) -> None:
        for category in ("feedback", "local"):
            category_dir = args.client_dir / "generated" / category
            if not category_dir.exists():
                continue
            for round_dir in sorted(category_dir.glob("round_*")):
                try:
                    round_index = int(round_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if round_index > until_round:
                    continue
                for class_file in round_dir.glob("class_*/headlines.json"):
                    try:
                        label = int(class_file.parent.name.split("_")[1])
                        with class_file.open(encoding="utf-8") as handle:
                            headlines = json.load(handle)
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(headlines, list):
                        continue
                    valid = [str(item) for item in headlines if isinstance(item, str) and item.strip()]
                    texts.extend(valid)
                    labels.extend([label] * len(valid))

    load_past_synthetic(last_round)
    print(
        f"[Client {args.client_id}] Local dataset contains {len(texts)} rows after resume."
    )

    def make_loaders() -> tuple[Any, Any]:
        encoded = classifier_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )

        class NewsDataset(Dataset):
            def __len__(self) -> int:
                return len(labels)

            def __getitem__(self, index: int) -> Mapping[str, Any]:
                return {
                    "input_ids": encoded["input_ids"][index],
                    "attention_mask": encoded["attention_mask"][index],
                    "labels": torch.tensor(labels[index], dtype=torch.long),
                }

        dataset = NewsDataset()
        train_length = max(1, int(0.8 * len(dataset)))
        validation_length = len(dataset) - train_length
        if validation_length == 0:
            train_length -= 1
            validation_length = 1
        train_data, validation_data = random_split(
            dataset,
            [train_length, validation_length],
            generator=torch.Generator().manual_seed(args.seed),
        )
        if args.balance_data:
            class_counts = Counter(labels[index] for index in train_data.indices)
            weights = torch.tensor(
                [1.0 / class_counts[labels[index]] for index in train_data.indices],
                dtype=torch.float,
            )
            sampler = WeightedRandomSampler(
                weights, num_samples=len(train_data), replacement=True
            )
            train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(
            validation_data, batch_size=args.batch_size, shuffle=False
        )
        return train_loader, validation_loader

    train_loader, validation_loader = make_loaders()
    generator = HeadlineGenerator() if args.generator == "gemma" else None
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    def validate_feedback_from_server(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
        raw_payload = config.get("feedback_payload")
        if not raw_payload:
            return None
        try:
            payload = json.loads(str(raw_payload))
        except json.JSONDecodeError as exc:
            raise ValueError("Server feedback payload is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Server feedback payload must be a JSON object.")
        advertised_protocol = str(config.get("feedback_protocol", payload.get("protocol", "")))
        if advertised_protocol != args.feedback_protocol:
            raise ValueError(
                f"Server protocol {advertised_protocol!r} does not match client protocol {args.feedback_protocol!r}."
            )
        try:
            return validate_feedback_payload(
                payload,
                protocol=advertised_protocol,
                allow_raw_feedback=args.allow_raw_feedback,
                domain="text",
            )
        except FeedbackProtocolError as exc:
            raise ValueError(f"Rejected server feedback payload: {exc}") from exc

    def generate_from_feedback(payload: Mapping[str, Any], round_index: int) -> int:
        if generator is None:
            print(
                f"[Client {args.client_id}] Feedback arrived but generation is disabled."
            )
            return 0
        protocol = str(payload["protocol"])
        weak_labels = [int(label) for label in payload.get("weak_labels", [])]
        generated_by_label: dict[int, list[str]] = defaultdict(list)
        if protocol == "abstract":
            salient_positions = list(payload.get("salient_positions", []))
            for label in weak_labels:
                label_name = label_names.get(label, str(label))
                prompts = [
                    build_abstract_prompt(label_name, salient_positions)
                    for _ in range(args.generated_per_weak_label)
                ]
                generated_by_label[label].extend(generator.generate(prompts))
        elif protocol == "legacy":
            for item in payload.get("token_feedback", []):
                label = int(item["label"])
                label_name = label_names.get(label, str(label))
                tokens = [str(token) for token in item.get("tokens", [])]
                prompt = build_legacy_prompt(label_name, tokens)
                generated_by_label[label].extend(generator.generate([prompt]))
        else:
            raise ValueError(f"Unsupported normalized feedback protocol: {protocol}")

        total = 0
        destination = args.client_dir / "generated" / "feedback" / f"round_{round_index}"
        for label, headlines in generated_by_label.items():
            unique_headlines = list(dict.fromkeys(headline for headline in headlines if headline.strip()))
            if not unique_headlines:
                continue
            _write_json(destination / f"class_{label}" / "headlines.json", unique_headlines)
            texts.extend(unique_headlines)
            labels.extend([label] * len(unique_headlines))
            total += len(unique_headlines)

        receipt = {
            "round": round_index,
            "protocol": protocol,
            "weak_labels": weak_labels,
            "salient_position_count": len(payload.get("salient_positions", [])),
            "legacy_token_group_count": len(payload.get("token_feedback", [])),
            "generated_count": total,
        }
        _write_json(args.client_dir / "audit" / f"feedback_receipt_round_{round_index}.json", receipt)
        return total

    def generate_local_baseline(round_index: int) -> int:
        if generator is None or args.local_gen_ratio == 0.0:
            return 0
        amount = int(len(texts) * args.local_gen_ratio)
        if amount == 0:
            return 0
        generated_by_label: dict[int, list[str]] = defaultdict(list)
        labels_for_generation = random.choices(labels, k=amount)
        for label in labels_for_generation:
            label_name = label_names.get(label, str(label))
            prompt = build_abstract_prompt(label_name, [])
            generated_by_label[label].extend(generator.generate([prompt]))
        total = 0
        destination = args.client_dir / "generated" / "local" / f"round_{round_index}"
        for label, headlines in generated_by_label.items():
            unique_headlines = list(dict.fromkeys(headline for headline in headlines if headline.strip()))
            if not unique_headlines:
                continue
            _write_json(destination / f"class_{label}" / "headlines.json", unique_headlines)
            texts.extend(unique_headlines)
            labels.extend([label] * len(unique_headlines))
            total += len(unique_headlines)
        return total

    class NewsClient(fl.client.NumPyClient):
        def get_parameters(self, config: Mapping[str, Any] | None = None) -> list[Any]:
            return get_parameters(model)

        def fit(self, parameters: Sequence[Any], config: Mapping[str, Any]) -> tuple[list[Any], int, dict[str, Any]]:
            nonlocal train_loader, validation_loader
            global_round = int(config.get("global_round", config.get("server_round", 1)))
            expected_label_map = str(config.get("label_map_fingerprint", ""))
            expected_num_labels = int(config.get("num_labels", -1))
            if not expected_label_map:
                raise ValueError("Server did not provide a label-map fingerprint.")
            if expected_label_map != label_map_fingerprint or expected_num_labels != num_labels:
                raise ValueError(
                    "Client label map does not match the server label map; "
                    "supply the server-generated --label-map file to every client."
                )
            self.set_parameters(parameters)
            payload = validate_feedback_from_server(config)
            generated_count = 0
            if payload is not None:
                generated_count = generate_from_feedback(payload, global_round)
            elif args.local_gen_ratio > 0.0:
                generated_count = generate_local_baseline(global_round)
            if generated_count:
                train_loader, validation_loader = make_loaders()
                if args.save_merged_data:
                    snapshot = args.client_dir / "merged_data" / f"round_{global_round}.csv"
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"text": texts, "label": labels}).to_csv(snapshot, index=False)
                print(
                    f"[Client {args.client_id}] Added {generated_count} local synthetic rows at round {global_round}."
                )

            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
            if args.balance_data:
                counts = Counter(labels)
                class_weights = torch.ones(num_labels, dtype=torch.float)
                for label, count in counts.items():
                    if label < num_labels and count > 0:
                        class_weights[label] = 1.0 / count
                loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
            else:
                loss_function = nn.CrossEntropyLoss()
            loss_total = 0.0
            correct_total = 0
            sample_total = 0
            for _epoch in range(args.epochs):
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_tensor = batch["labels"].to(device)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        logits = model(input_ids, attention_mask=attention_mask).logits
                        loss = loss_function(logits, labels_tensor)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loss_total += float(loss.detach().item()) * labels_tensor.size(0)
                    correct_total += int((logits.argmax(dim=1) == labels_tensor).sum().item())
                    sample_total += int(labels_tensor.size(0))
            train_accuracy = correct_total / sample_total if sample_total else 0.0
            train_loss = loss_total / sample_total if sample_total else 0.0
            save_checkpoint(
                global_round,
                {
                    "client_id": args.client_id,
                    "train_accuracy": train_accuracy,
                    "generated_count": generated_count,
                    "feedback_protocol": str(payload.get("protocol")) if payload else "none",
                },
            )
            return self.get_parameters(), len(train_loader.dataset), {
                "client_id": args.client_id,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "generated_count": generated_count,
                "feedback_received": int(payload is not None),
            }

        def evaluate(self, parameters: Sequence[Any], config: Mapping[str, Any]) -> tuple[float, int, dict[str, Any]]:
            self.set_parameters(parameters)
            model.eval()
            correct = 0
            total = 0
            loss_total = 0.0
            loss_function = nn.CrossEntropyLoss()
            with torch.no_grad():
                for batch in validation_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_tensor = batch["labels"].to(device)
                    logits = model(input_ids, attention_mask=attention_mask).logits
                    loss_total += float(loss_function(logits, labels_tensor).item()) * labels_tensor.size(0)
                    correct += int((logits.argmax(dim=1) == labels_tensor).sum().item())
                    total += int(labels_tensor.size(0))
            accuracy = correct / total if total else 0.0
            loss = loss_total / total if total else 0.0
            return loss, total, {"client_id": args.client_id, "accuracy": accuracy}

        def set_parameters(self, parameters: Sequence[Any]) -> None:
            set_parameters(model, parameters)

    print(
        f"[Client {args.client_id}] Starting against {args.server_address}; "
        f"protocol={args.feedback_protocol}, generator={args.generator}, resume_round={last_round}."
    )
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=NewsClient(),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    try:
        args = validate_startup_args(parser.parse_args(argv))
        run_client(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
