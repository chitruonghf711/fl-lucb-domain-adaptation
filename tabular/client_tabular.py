"""Federated learning client for the tabular churn task (Flower + PyTorch + SDV).

Each client trains a small MLP on its private CSV shard. The default feedback
protocol sends only weak class labels and aggregate SHAP feature ranks. The
client generates class-targeted TVAE samples locally and prioritizes sparse
regions of its own ranked features. The historical masked-value pathway remains
available only through explicit legacy raw-feedback flags.

Run one process per client, each with its own id and data shard:

    python client_tabular.py --client-id client1 --train-csv dataset/7030/train1.csv \
        --server-address 127.0.0.1:9000
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer
    HAS_SDV = True
except ImportError:
    HAS_SDV = False
    SingleTableMetadata = Any  # type: ignore[assignment, misc]
    TVAESynthesizer = Any  # type: ignore[assignment, misc]
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl_lucb import validate_feedback_payload

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
# Values that identify this client (id, data shard, server address) are CLI flags;
# everything below is experiment tuning and is edited here.
CONFIG = {
    "client_id": "client1",  # Default id, overridden by --client-id
    "train_csv_path": "dataset/7030/train1.csv",  # Default shard, overridden by --train-csv
    "server_address": "127.0.0.1:9000",  # Default address, overridden by --server-address
    "target_column": "Exited",  # Label column in the training CSV
    "log_dir": "logs/lucb_pilot",  # Run log folder, overridden by --log-dir;
                              # this client writes to <log_dir>/clients/<client-id>/
    "gen_model_path": "tvae_gen.pkl",  # Path to the pre-trained TVAE generator
    "epoch": 50,  # Number of training epochs
    "batch_size": 32,  # Mini-batch size for training
    "learning_rate": 0.001,  # Adam learning rate
    "gen_multiplier": 3.0,  # Synthetic data size = multiplier x original size
    "cosine_sim_threshold": 0.9,  # Cosine similarity threshold for filtering
    "max_gen_attempts": 10,  # Max generate-and-filter rounds before giving up
    "use_feedback": True,  # True = generate based on server feedback; False = ignore feedback
    "gen_multiplier_fixed": 0.1,  # Synthetic data ratio ( use_feedback = False)
                                  # Final Synthetic data ( use_feedback = True)
    "feedback_protocol": "abstract",  # Default protocol; legacy requires an explicit raw-data acknowledgement
    "allow_raw_feedback": False,  # Required before this client accepts legacy server payloads
    "smoke_no_generation": False,  # Test-only: validate feedback while skipping TVAE work
}


# === CLI & LOGGING ===
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--client-id", default=CONFIG["client_id"],
                        help="Unique client id; the server keys its feedback on it")
    parser.add_argument("--train-csv", default=CONFIG["train_csv_path"],
                        help="Path to this client's local training CSV")
    parser.add_argument("--server-address", default=CONFIG["server_address"],
                        help="Address of the Flower server, e.g. 127.0.0.1:9000")
    parser.add_argument("--log-dir", default=CONFIG["log_dir"],
                        help="Run log folder; use the same one as the server, e.g. logs/lucb_pilot")
    parser.add_argument("--feedback-protocol", choices=("abstract", "legacy"),
                        default=CONFIG["feedback_protocol"],
                        help="Expected server feedback protocol; abstract is the privacy-preserving default")
    parser.add_argument("--allow-raw-feedback", action="store_true",
                        help="Required acknowledgement before accepting legacy raw-value feedback")
    parser.add_argument("--epochs", type=int, default=CONFIG["epoch"],
                        help="Local training epochs; use 1 for the lightweight smoke run")
    parser.add_argument("--smoke-no-generation", action="store_true",
                        help="Test-only: skip TVAE setup and generation but validate received feedback")
    return parser.parse_args()


def setup_logging(log_dir: str) -> str:
    """Log to a timestamped file inside log_dir and return its path."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # SDV reports every fit/sample call on this logger; keep it out of the client log.
    logging.getLogger("SingleTableSynthesizer").propagate = False
    return log_path


# === MODEL DEFINITION ===
class SimpleNN(nn.Module):
    """Same architecture as the server's global model; weights are exchanged as-is."""

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# === TVAE SYNTHESIZERS ===
def fit_restore_synthesizer(df: pd.DataFrame) -> TVAESynthesizer:
    """Fit a TVAE on the local data; used to fill the features masked by SHAP."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(df)
    return synthesizer


def load_generator(path: str) -> TVAESynthesizer:
    """Unpickle the pre-trained TVAE generator.

    tvae_gen.pkl was saved from an environment with numpy >= 2.0, which pickles
    RandomState / BitGenerator objects in a format numpy 1.x cannot rebuild
    (torch 2.2 pins numpy < 2). Temporarily patch numpy's reconstructors so the
    legacy random state is accepted, then restore them.
    """
    import numpy.random as npr
    import numpy.random._pickle as np_pickle
    from numpy.random import BitGenerator, Generator, RandomState

    def _setstate(self, state):
        self.state = state[0] if isinstance(state, tuple) else state

    def _bit_generator(bit_generator=None, **kwargs):
        if isinstance(bit_generator, BitGenerator):
            return bit_generator
        base = bit_generator if isinstance(bit_generator, type) else getattr(
            npr, bit_generator if isinstance(bit_generator, str) else "MT19937")
        # Keep the original class name: numpy validates state["bit_generator"] against it
        return type(base.__name__, (base,), {"__setstate__": _setstate})()

    saved = (np_pickle.__bit_generator_ctor,
             np_pickle.__randomstate_ctor,
             np_pickle.__generator_ctor)
    np_pickle.__bit_generator_ctor = _bit_generator
    np_pickle.__randomstate_ctor = lambda bg=None, **kw: RandomState(_bit_generator(bg))
    np_pickle.__generator_ctor = lambda bg=None, **kw: Generator(_bit_generator(bg))
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        (np_pickle.__bit_generator_ctor,
         np_pickle.__randomstate_ctor,
         np_pickle.__generator_ctor) = saved


# === FEEDBACK HANDLING ===
def feedback_to_dataframe(shap_feedback: List[Dict[str, Any]], target_column: str) -> pd.DataFrame:
    """Turn the server's [{features, label}, ...] payload into a DataFrame."""
    return pd.DataFrame([{**item["features"], target_column: item["label"]} for item in shap_feedback])


def normalized_abstract_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate and extract abstract tabular feedback without accepting row values."""

    normalized = validate_feedback_payload(
        payload, protocol="abstract", allow_raw_feedback=False, domain="tabular"
    )
    ranking = list(normalized.get("feature_ranking", []))
    return [
        {"label": label, "feature_ranks": ranking}
        for label in normalized.get("weak_labels", [])
    ]


def ranked_feature_names(item: Dict[str, Any], available_columns: List[str]) -> List[str]:
    """Return safe feature names ordered by server-side aggregate SHAP rank."""

    names = []
    for value in item.get("feature_ranks", []):
        name = value.get("feature") if isinstance(value, dict) else value
        if isinstance(name, str) and name in available_columns and name not in names:
            names.append(name)
    return names


def local_sparsity_scores(candidates: pd.DataFrame, local_df: pd.DataFrame,
                          feature_names: List[str]) -> pd.Series:
    """Score generated candidates by sparse local regions of ranked features.

    This uses only the client shard and TVAE-generated rows. It receives no
    feature values from the server, so abstract feedback remains row-free.
    """

    scores = pd.Series(0.0, index=candidates.index, dtype=float)
    for rank, feature in enumerate(feature_names, start=1):
        if feature not in candidates.columns or feature not in local_df.columns:
            continue
        weight = 1.0 / rank
        local_values = local_df[feature].dropna()
        if local_values.empty:
            continue
        numeric_local = pd.to_numeric(local_values, errors="coerce")
        numeric_candidates = pd.to_numeric(candidates[feature], errors="coerce")
        if numeric_local.notna().all() and numeric_candidates.notna().any():
            lower, upper = numeric_local.quantile([0.01, 0.99]).tolist()
            if lower == upper:
                continue
            counts, edges = np.histogram(numeric_local.to_numpy(), bins=10, range=(lower, upper))
            bins = np.clip(np.searchsorted(edges, numeric_candidates.to_numpy(), side="right") - 1,
                           0, len(counts) - 1)
            scores += weight / (1.0 + counts[bins])
        else:
            frequencies = local_values.astype(str).value_counts(normalize=True)
            scores += candidates[feature].astype(str).map(
                lambda value: 1.0 - float(frequencies.get(value, 0.0))
            ).fillna(1.0) * weight
    return scores


def collect_abstract_synthetic_samples(generator: TVAESynthesizer, local_df: pd.DataFrame,
                                       items: List[Dict[str, Any]], base_size: int) -> pd.DataFrame:
    """Generate class-targeted samples and prefer sparse local ranked-feature regions."""

    required = int(CONFIG["gen_multiplier_fixed"] * base_size)
    if required <= 0 or not items:
        return pd.DataFrame(columns=local_df.columns)
    target_column = CONFIG["target_column"]
    per_item = max(1, int(np.ceil(required / len(items))))
    selected_frames = []
    for item in items:
        label = item.get("label")
        feature_names = ranked_feature_names(
            item, [column for column in local_df.columns if column != target_column]
        )
        selected = pd.DataFrame(columns=local_df.columns)
        for _ in range(CONFIG["max_gen_attempts"]):
            generated = generator.sample(int(CONFIG["gen_multiplier"] * base_size))
            if target_column not in generated.columns:
                logging.warning("Generated data did not contain the target column for abstract feedback")
                break
            same_label = generated[generated[target_column].astype(str) == str(label)].copy()
            if same_label.empty:
                continue
            same_label["__local_sparsity_score"] = local_sparsity_scores(
                same_label, local_df, feature_names
            )
            selected = pd.concat([selected, same_label], ignore_index=True)
            selected = selected.drop_duplicates().sort_values(
                "__local_sparsity_score", ascending=False
            ).head(per_item)
            if len(selected) >= per_item:
                break
        if not selected.empty:
            selected_frames.append(selected.drop(columns=["__local_sparsity_score"], errors="ignore"))
    if not selected_frames:
        return pd.DataFrame(columns=local_df.columns)
    return pd.concat(selected_frames, ignore_index=True).drop_duplicates().head(required)


def restore_missing_values(fb_df: pd.DataFrame, synthesizer: TVAESynthesizer) -> pd.DataFrame:
    """Fill the features the server masked out with TVAE-sampled values."""
    restored_df = fb_df.copy()
    for col in restored_df.columns:
        if restored_df[col].isna().sum() > 0:
            fill_values = synthesizer.sample(len(restored_df))[col]
            restored_df[col] = restored_df[col].fillna(fill_values)
    return restored_df


def match_by_masked_features(gen_df: pd.DataFrame, fb_df: pd.DataFrame,
                             cosine_threshold: float) -> pd.DataFrame:
    """Keep the generated rows that are cosine-similar to any feedback row.

    Only the features present (non-NaN) in a feedback row take part in the
    comparison, so partially masked feedback can still be matched.
    """
    matched_rows = []
    for _, row in fb_df.iterrows():
        mask = ~row.isna()  # Mask of present (non-missing) features
        if mask.sum() == 0:
            continue
        fb_vec = row[mask].values.reshape(1, -1)
        gen_filtered = gen_df[mask.index[mask]]
        sims = cosine_similarity(gen_filtered.values, fb_vec)[:, 0]
        selected = gen_df[sims >= cosine_threshold]
        matched_rows.extend(selected.to_dict(orient="records"))
    return pd.DataFrame(matched_rows)


def collect_synthetic_samples(generator: TVAESynthesizer, restored_df: pd.DataFrame,
                              base_size: int) -> pd.DataFrame:
    """Generate and filter repeatedly until enough samples match the feedback."""
    required = int(CONFIG["gen_multiplier_fixed"] * base_size)
    batch_size = int(CONFIG["gen_multiplier"] * base_size)
    total_selected = pd.DataFrame()

    for attempt in range(CONFIG["max_gen_attempts"]):  # Avoid infinite loop
        gen_batch = generator.sample(batch_size)
        filtered = match_by_masked_features(gen_batch, restored_df, CONFIG["cosine_sim_threshold"])
        total_selected = pd.concat([total_selected, filtered]).drop_duplicates().reset_index(drop=True)

        logging.info(f"[GEN-{attempt + 1}] Filtered: {len(filtered)} | Total Selected: {len(total_selected)}")
        if len(total_selected) >= required:
            break

    selected_samples = total_selected.drop_duplicates().head(required)  # Truncate to exact required number
    logging.info(f"[FINAL] Selected {len(selected_samples)} synthetic samples (required {required})")
    return selected_samples


# === LOCAL TRAINING ===
def train_local_model(model: nn.Module, train_df: pd.DataFrame) -> int:
    """Train the model on train_df in place and return the number of samples used."""
    features = train_df.drop(columns=[CONFIG["target_column"]])
    labels = train_df[CONFIG["target_column"]]

    x_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(x_tensor, y_tensor),
                              batch_size=CONFIG["batch_size"], shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    epochs = CONFIG["epoch"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logging.info(f"Epoch {epoch + 1} / {epochs} - Loss: {avg_loss:.4f}")
            print(f"[CLIENT] Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

    return len(x_tensor)


# === FLOWER CLIENT DEFINITION ===
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model: nn.Module, train_df: pd.DataFrame,
                 restore_synthesizer: Optional[TVAESynthesizer], generator: Optional[TVAESynthesizer],
                 client_log_dir: Optional[str] = None):
        self.client_id = client_id
        self.model = model
        self.train_df = train_df
        self.restore_synthesizer = restore_synthesizer
        self.generator = generator
        # Rebuilt from train_df at every fit() call, so restored/synthetic rows never
        # accumulate across rounds; it is kept as state only to report the last size.
        self.train_data = train_df.copy()
        self.client_log_dir = client_log_dir or CONFIG.get("client_log_dir", f"logs/clients/{client_id}")
        self.time_logs: List[Dict[str, Any]] = []
        self.round_counter = 0

    def get_parameters(self, config: Dict[str, Any]) -> List[Any]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[Any]) -> None:
        state_dict = self.model.state_dict()
        for key, value in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(value)
        self.model.load_state_dict(state_dict)

    def save_time_summary(self) -> None:
        """Persist per-round client execution timing to CSV."""
        if not self.time_logs:
            return
        os.makedirs(self.client_log_dir, exist_ok=True)
        csv_path = os.path.join(self.client_log_dir, "client_time_summary.csv")
        pd.DataFrame(self.time_logs).to_csv(csv_path, index=False)

    def fit(self, parameters: List[Any], config: Dict[str, Any]) -> Tuple[List[Any], int, Dict[str, Any]]:
        t_fit_start = time.time()
        self.round_counter += 1
        current_round = config.get("server_round", self.round_counter)
        restore_time = 0.0
        gen_time = 0.0
        gen_samples_count = 0

        self.set_parameters(parameters)

        protocol = str(config.get("feedback_protocol", CONFIG["feedback_protocol"]))
        raw_feedback = json.loads(config.get("feedback", "[]"))
        restored_df = pd.DataFrame()
        selected_samples = pd.DataFrame()

        if protocol not in {"abstract", "legacy"}:
            raise ValueError(f"Unsupported feedback protocol from server: {protocol}")
        if protocol == "legacy" and not CONFIG["allow_raw_feedback"]:
            raise ValueError("Legacy feedback was received without --allow-raw-feedback")

        if CONFIG["smoke_no_generation"]:
            if raw_feedback and protocol == "abstract":
                items = normalized_abstract_items(raw_feedback)
                logging.info("Smoke mode validated %d abstract weak class summaries", len(items))
            elif raw_feedback:
                validate_feedback_payload(
                    {
                        "protocol": "legacy", "domain": "tabular",
                        "weak_labels": [], "shap_feedback": raw_feedback,
                        "reference": json.loads(config.get("reference", "[]")),
                    },
                    protocol="legacy", allow_raw_feedback=True, domain="tabular"
                )
                logging.info("Smoke mode validated explicit legacy feedback")
            self.train_data = self.train_df.copy()
            logging.info("Smoke mode skipped TVAE generation and trained only on the local shard")
        elif CONFIG["use_feedback"] and raw_feedback:
            if self.restore_synthesizer is None or self.generator is None:
                raise RuntimeError("TVAE synthesizers are required outside --smoke-no-generation mode")
            if protocol == "abstract":
                items = normalized_abstract_items(raw_feedback)
                logging.info("Using abstract feedback for %d weak class summaries", len(items))
                t_gen = time.time()
                selected_samples = collect_abstract_synthetic_samples(
                    self.generator, self.train_df, items, len(self.train_df)
                )
                gen_time = time.time() - t_gen
                gen_samples_count = len(selected_samples)
                self.train_data = pd.concat([self.train_df, selected_samples], ignore_index=True)
                print(f"[CLIENT] Abstract feedback selected {len(selected_samples)} local synthetic samples")
            else:
                legacy_payload = validate_feedback_payload(
                    {
                        "protocol": "legacy", "domain": "tabular",
                        "weak_labels": [], "shap_feedback": raw_feedback,
                        "reference": json.loads(config.get("reference", "[]")),
                    },
                    protocol="legacy", allow_raw_feedback=True, domain="tabular"
                )
                shap_feedback = legacy_payload.get("shap_feedback", [])
                logging.warning("Using explicit legacy raw-value feedback for %d samples", len(shap_feedback))
                fb_df = feedback_to_dataframe(shap_feedback, CONFIG["target_column"])
                t_res = time.time()
                restored_df = restore_missing_values(fb_df, self.restore_synthesizer)
                restore_time = time.time() - t_res
                logging.info("NaN before restore: %d | after restore: %d",
                             fb_df.isna().sum().sum(), restored_df.isna().sum().sum())
                t_gen = time.time()
                selected_samples = collect_synthetic_samples(self.generator, restored_df, len(self.train_df))
                gen_time = time.time() - t_gen
                gen_samples_count = len(selected_samples)
                self.train_data = pd.concat([self.train_df, restored_df, selected_samples], ignore_index=True)
                print(f"[CLIENT] Legacy feedback selected {len(selected_samples)} synthetic samples")
        else:
            logging.info("Generating fixed synthetic data without feedback")
            if self.generator is None:
                raise RuntimeError("TVAE generator is required outside --smoke-no-generation mode")

            num_gen = int(CONFIG["gen_multiplier_fixed"] * len(self.train_df))
            t_gen = time.time()
            gen_fixed = self.generator.sample(num_gen)
            gen_time = time.time() - t_gen
            gen_samples_count = len(gen_fixed)
            logging.info("Generated %d fixed synthetic samples", num_gen)
            self.train_data = pd.concat([self.train_df, gen_fixed], ignore_index=True)

        logging.info(f"Final train size: Original={len(self.train_df)} | Restored={len(restored_df)} "
                     f"| Synthetic={len(selected_samples)} | Total={len(self.train_data)}")

        t_train = time.time()
        num_examples = train_local_model(self.model, self.train_data)
        train_time = time.time() - t_train
        total_fit_time = time.time() - t_fit_start

        # Record client round execution metrics
        self.time_logs.append({
            "round": current_round,
            "total_fit_time_sec": round(total_fit_time, 2),
            "restore_time_sec": round(restore_time, 2),
            "gen_time_sec": round(gen_time, 2),
            "train_time_sec": round(train_time, 2),
            "train_samples": num_examples,
            "gen_samples": gen_samples_count,
        })
        self.save_time_summary()

        logging.info("=== ROUND COMPLETED ===\n")
        return self.get_parameters(config), num_examples, {"client_id": self.client_id}

        num_examples = train_local_model(self.model, self.train_data)
        logging.info("=== ROUND COMPLETED ===\n")
        return self.get_parameters(config), num_examples, {"client_id": self.client_id}

    def evaluate(self, parameters: List[Any], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        # Dummy evaluation: the server evaluates centrally and skips client evaluation
        return 0.0, len(self.train_data), {}


# === MAIN ===
def main() -> None:
    args = parse_args()
    CONFIG["client_id"] = args.client_id
    CONFIG["train_csv_path"] = args.train_csv
    CONFIG["server_address"] = args.server_address
    CONFIG["log_dir"] = args.log_dir
    CONFIG["feedback_protocol"] = args.feedback_protocol
    CONFIG["allow_raw_feedback"] = bool(args.allow_raw_feedback)
    CONFIG["epoch"] = args.epochs
    CONFIG["smoke_no_generation"] = bool(args.smoke_no_generation)
    if CONFIG["epoch"] <= 0:
        raise ValueError("--epochs must be positive")
    if CONFIG["feedback_protocol"] == "legacy" and not CONFIG["allow_raw_feedback"]:
        raise ValueError("--feedback-protocol legacy requires --allow-raw-feedback")
    if CONFIG["feedback_protocol"] == "legacy":
        print("[CLIENT] WARNING: Legacy feedback can contain target-domain raw values.")
    if CONFIG["smoke_no_generation"]:
        print("[CLIENT] Smoke mode enabled: TVAE setup and generation are skipped.")
    CONFIG["client_log_dir"] = os.path.join(args.log_dir, "clients", args.client_id)

    log_path = setup_logging(CONFIG["client_log_dir"])
    logging.info("=== CLIENT STARTED ===")
    logging.info(f"CONFIG: {CONFIG}")
    print(f"[CLIENT] {CONFIG['client_id']} | logging to {log_path}")

    print(f"[CLIENT] Loading training data from {CONFIG['train_csv_path']}...")
    train_df = pd.read_csv(CONFIG["train_csv_path"])

    restore_synthesizer: Optional[TVAESynthesizer] = None
    generator: Optional[TVAESynthesizer] = None
    if not CONFIG["smoke_no_generation"]:
        print("[CLIENT] Fitting the TVAE used to restore masked feedback...")
        restore_synthesizer = fit_restore_synthesizer(train_df)
        print(f"[CLIENT] Loading the pre-trained generator from {CONFIG['gen_model_path']}...")
        generator = load_generator(CONFIG["gen_model_path"])

    model = SimpleNN(train_df.drop(columns=[CONFIG["target_column"]]).shape[1])
    client = FLClient(
        CONFIG["client_id"], model, train_df, restore_synthesizer, generator,
        client_log_dir=CONFIG["client_log_dir"]
    )

    print(f"[CLIENT] Connecting to {CONFIG['server_address']}...")
    fl.client.start_numpy_client(server_address=CONFIG["server_address"], client=client)


if __name__ == "__main__":
    main()
