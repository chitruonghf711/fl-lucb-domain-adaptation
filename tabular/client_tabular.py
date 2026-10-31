"""Federated learning client for tabular churn task (Flower + PyTorch + SDV).

Each client trains a small MLP on its private CSV shard. When receiving server feedback,
it adapts its local distribution using TVAE synthetic data generation.

Feedback Protocols & Target Leakage Prevention:
-----------------------------------------------
1. "abstract" protocol (default):
   - The server only sends metadata: weak class label and top-k important feature names.
   - ZERO test samples or feature values are transmitted from the server.
   - The client generates synthetic data conditioned on the weak class and trains on:
     `self.train_data = pd.concat([self.train_df, selected_samples], ignore_index=True)`
   - Completely privacy-preserving with ZERO test data leakage.

2. "detailed" protocol (legacy):
   - The server sends masked test samples with top features kept and other features as NaN.
   - The client restores missing features with its local TVAE (`restored_df`) and uses them
     to guide similarity filtering of generated synthetic samples (`selected_samples`).
   - CRITICAL PRIVACY FIX:
     - By default (`allow_target_leakage=False`), `restored_df` is NOT concatenated into
       the training data (`self.train_data = pd.concat([self.train_df, selected_samples])`).
     - Only if `--allow-target-leakage` is explicitly passed (for reproducing legacy leakage)
       will `restored_df` be concatenated.

Usage:
    python client_tabular.py --client-id client1 --train-csv dataset/7030/train1.csv \
        --server-address 127.0.0.1:9000
"""

import argparse
import json
import logging
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer
    HAS_SDV = True
except ImportError:
    HAS_SDV = False

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
CONFIG = {
    "client_id": "client1",
    "train_csv_path": "dataset/7030/train1.csv",
    "server_address": "127.0.0.1:9000",
    "target_column": "Exited",
    "log_dir": "logs/exp_minibatch",
    "gen_model_path": "tvae_gen.pkl",
    "epoch": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "gen_multiplier": 3.0,
    "cosine_sim_threshold": 0.9,
    "max_gen_attempts": 10,
    "use_feedback": True,
    "gen_multiplier_fixed": 0.1,
    # Privacy / Data leakage safeguard:
    # Set False to guarantee restored test-domain samples are NOT fed into training!
    "allow_target_leakage": False,
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
                        help="Run log folder; e.g. logs/exp_minibatch")
    parser.add_argument("--gen-model-path", default=CONFIG["gen_model_path"],
                        help="Path to pre-trained TVAE generator pickle")
    parser.add_argument("--allow-target-leakage", action="store_true", default=False,
                        help="DANGEROUS: Concatenate restored server test samples into local training data (reproduces legacy leakage)")
    parser.add_argument("--epoch", type=int, default=CONFIG["epoch"],
                        help="Local training epochs per round")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"],
                        help="Local training batch size")
    parser.add_argument("--learning-rate", type=float, default=CONFIG["learning_rate"],
                        help="Learning rate for Adam optimizer")
    return parser.parse_args()


def setup_logging(log_dir: str) -> str:
    """Log to a timestamped file inside log_dir and return its path."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
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
def fit_restore_synthesizer(df: pd.DataFrame) -> Any:
    """Fit a TVAE on the local data; used to fill features in detailed feedback."""
    if not HAS_SDV:
        return None
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(df)
    return synthesizer


def load_generator(path: str) -> Any:
    """Unpickle the pre-trained TVAE generator with numpy compatibility patch."""
    if not os.path.exists(path):
        return None
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


def restore_missing_values(fb_df: pd.DataFrame, synthesizer: Any) -> pd.DataFrame:
    """Fill the features the server masked out with TVAE-sampled values."""
    if synthesizer is None:
        return fb_df.fillna(0.0)
    restored_df = fb_df.copy()
    for col in restored_df.columns:
        if restored_df[col].isna().sum() > 0:
            fill_values = synthesizer.sample(len(restored_df))[col]
            restored_df[col] = restored_df[col].fillna(fill_values)
    return restored_df


def match_by_masked_features(gen_df: pd.DataFrame, fb_df: pd.DataFrame,
                             cosine_threshold: float) -> pd.DataFrame:
    """Keep generated rows that are cosine-similar to any feedback row."""
    matched_rows = []
    for _, row in fb_df.iterrows():
        mask = ~row.isna()
        if mask.sum() == 0:
            continue
        fb_vec = row[mask].values.reshape(1, -1)
        gen_filtered = gen_df[mask.index[mask]]
        sims = cosine_similarity(gen_filtered.values, fb_vec)[:, 0]
        selected = gen_df[sims >= cosine_threshold]
        matched_rows.extend(selected.to_dict(orient="records"))
    return pd.DataFrame(matched_rows)


def collect_synthetic_samples(generator: Any, restored_df: pd.DataFrame,
                              base_size: int) -> pd.DataFrame:
    """Generate and filter repeatedly until enough samples match the feedback."""
    required = int(CONFIG["gen_multiplier_fixed"] * base_size)
    if generator is None:
        return pd.DataFrame()
    batch_size = int(CONFIG["gen_multiplier"] * base_size)
    total_selected = pd.DataFrame()

    for attempt in range(CONFIG["max_gen_attempts"]):
        gen_batch = generator.sample(batch_size)
        filtered = match_by_masked_features(gen_batch, restored_df, CONFIG["cosine_sim_threshold"])
        total_selected = pd.concat([total_selected, filtered]).drop_duplicates().reset_index(drop=True)

        logging.info(f"[GEN-{attempt + 1}] Filtered: {len(filtered)} | Total Selected: {len(total_selected)}")
        if len(total_selected) >= required:
            break

    selected_samples = total_selected.drop_duplicates().head(required)
    logging.info(f"[FINAL] Selected {len(selected_samples)} synthetic samples (required {required})")
    return selected_samples


# === LOCAL TRAINING ===
def train_local_model(model: nn.Module, train_df: pd.DataFrame) -> int:
    """Train the model on train_df in place and return the number of samples used."""
    features = train_df.drop(columns=[CONFIG["target_column"]])
    labels = train_df[CONFIG["target_column"]]

    x_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    epochs = CONFIG["epoch"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
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
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_df: pd.DataFrame,
        restore_synthesizer: Any,
        generator: Any,
        allow_target_leakage: bool = False
    ):
        self.client_id = client_id
        self.model = model
        self.train_df = train_df
        self.restore_synthesizer = restore_synthesizer
        self.generator = generator
        self.allow_target_leakage = allow_target_leakage
        self.train_data = train_df.copy()

    def get_parameters(self, config: Dict[str, Any]) -> List[Any]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[Any]) -> None:
        state_dict = self.model.state_dict()
        for key, value in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(value)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: List[Any], config: Dict[str, Any]) -> Tuple[List[Any], int, Dict[str, Any]]:
        self.set_parameters(parameters)

        feedback_raw = config.get("feedback", "{}")
        try:
            feedback_payload = json.loads(feedback_raw)
        except Exception:
            feedback_payload = {}

        # Fallback to legacy config keys if needed
        legacy_feedback = json.loads(config.get("legacy_feedback", "[]"))
        if not feedback_payload and legacy_feedback:
            feedback_payload = {"protocol": "detailed", "shap_feedback": legacy_feedback}

        restored_df = pd.DataFrame()
        selected_samples = pd.DataFrame()

        # Determine protocol
        protocol = feedback_payload.get("protocol")
        if not protocol:
            if isinstance(feedback_payload, list) or "shap_feedback" in feedback_payload:
                protocol = "detailed"
            elif "weak_label" in feedback_payload:
                protocol = "abstract"
            else:
                protocol = "none"

        has_valid_feedback = (
            (protocol == "abstract" and feedback_payload.get("weak_label") is not None) or
            (protocol == "detailed" and bool(feedback_payload.get("shap_feedback") or isinstance(feedback_payload, list)))
        )

        if CONFIG["use_feedback"] and has_valid_feedback:
            if protocol == "abstract":
                weak_label = feedback_payload.get("weak_label", 1)
                top_features = feedback_payload.get("top_features", [])
                logging.info(f"[ABSTRACT FEEDBACK] Weak label={weak_label}, top features={top_features}")
                print(f"[CLIENT] Received abstract feedback: Weak Label={weak_label}, Top Features={top_features}")

                required = int(CONFIG["gen_multiplier_fixed"] * len(self.train_df))
                batch_size = max(required * 2, int(CONFIG["gen_multiplier"] * len(self.train_df)))

                total_selected = pd.DataFrame()
                if self.generator is not None:
                    for _ in range(CONFIG["max_gen_attempts"]):
                        gen_batch = self.generator.sample(batch_size)
                        if CONFIG["target_column"] in gen_batch.columns:
                            matched = gen_batch[gen_batch[CONFIG["target_column"]] == weak_label]
                        else:
                            matched = gen_batch.copy()
                            matched[CONFIG["target_column"]] = weak_label
                        total_selected = pd.concat([total_selected, matched]).drop_duplicates().reset_index(drop=True)
                        if len(total_selected) >= required:
                            break

                selected_samples = total_selected.head(required) if not total_selected.empty else pd.DataFrame()
                logging.info(f"[ABSTRACT] Selected {len(selected_samples)} synthetic samples for weak class {weak_label}")
                print(f"[CLIENT] Generated {len(selected_samples)} synthetic samples for weak class {weak_label} (Target leakage: 0)")

                # ZERO test data leakage
                self.train_data = pd.concat([self.train_df, selected_samples], ignore_index=True)

            elif protocol == "detailed":
                shap_entries = feedback_payload.get(
                    "shap_feedback",
                    feedback_payload if isinstance(feedback_payload, list) else []
                )
                logging.info(f"[DETAILED FEEDBACK] Using masked feedback: {len(shap_entries)} samples")
                fb_df = feedback_to_dataframe(shap_entries, CONFIG["target_column"])

                # Restore missing features using TVAE
                restored_df = restore_missing_values(fb_df, self.restore_synthesizer)
                logging.info(f"NaN before restore: {fb_df.isna().sum().sum()} | after restore: {restored_df.isna().sum().sum()}")

                # Select synthetic samples cosine-similar to restored feedback
                selected_samples = collect_synthetic_samples(self.generator, restored_df, len(self.train_df))

                # PREVENT TARGET DOMAIN TEST DATA LEAKAGE:
                if self.allow_target_leakage:
                    logging.warning("[LEAKAGE ACTIVE] Restored test-domain samples are being added to training data!")
                    print(f"[WARNING] Concatenating {len(restored_df)} target test samples into training data (Data Leakage Mode)!")
                    self.train_data = pd.concat([self.train_df, restored_df, selected_samples], ignore_index=True)
                else:
                    logging.info(f"[LEAKAGE PREVENTED] {len(restored_df)} restored test samples excluded from training data.")
                    print(f"[CLIENT] Safe Mode: {len(restored_df)} restored test samples used ONLY for guidance, NOT concatenated into training set.")
                    self.train_data = pd.concat([self.train_df, selected_samples], ignore_index=True)
        else:
            logging.info("Generating fixed synthetic data without feedback")
            num_gen = int(CONFIG["gen_multiplier_fixed"] * len(self.train_df))
            gen_fixed = self.generator.sample(num_gen) if self.generator is not None else pd.DataFrame()
            self.train_data = pd.concat([self.train_df, gen_fixed], ignore_index=True)

        logging.info(f"Final train size: Original={len(self.train_df)} | Restored={len(restored_df)} "
                     f"| Synthetic={len(selected_samples)} | Total={len(self.train_data)}")

        num_examples = train_local_model(self.model, self.train_data)
        logging.info("=== ROUND COMPLETED ===\n")
        return self.get_parameters(config), num_examples, {"client_id": self.client_id}

    def evaluate(self, parameters: List[Any], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        return 0.0, len(self.train_data), {}


# === MAIN ===
def main() -> None:
    args = parse_args()
    CONFIG["client_id"] = args.client_id
    CONFIG["train_csv_path"] = args.train_csv
    CONFIG["server_address"] = args.server_address
    CONFIG["log_dir"] = args.log_dir
    CONFIG["gen_model_path"] = args.gen_model_path
    CONFIG["allow_target_leakage"] = args.allow_target_leakage
    CONFIG["epoch"] = args.epoch
    CONFIG["batch_size"] = args.batch_size
    CONFIG["learning_rate"] = args.learning_rate
    CONFIG["client_log_dir"] = os.path.join(args.log_dir, "clients", args.client_id)

    log_path = setup_logging(CONFIG["client_log_dir"])
    logging.info("=== CLIENT STARTED ===")
    logging.info(f"CONFIG: {CONFIG}")
    print(f"[CLIENT] {CONFIG['client_id']} | logging to {log_path}")
    print(f"[CLIENT] Target Data Leakage Mode: {CONFIG['allow_target_leakage']}")

    print(f"[CLIENT] Loading training data from {CONFIG['train_csv_path']}...")
    train_df = pd.read_csv(CONFIG["train_csv_path"])

    print("[CLIENT] Fitting the TVAE used to restore masked feedback...")
    restore_synthesizer = fit_restore_synthesizer(train_df)

    print(f"[CLIENT] Loading the pre-trained generator from {CONFIG['gen_model_path']}...")
    generator = load_generator(CONFIG["gen_model_path"])

    model = SimpleNN(train_df.drop(columns=[CONFIG["target_column"]]).shape[1])
    client = FLClient(
        client_id=CONFIG["client_id"],
        model=model,
        train_df=train_df,
        restore_synthesizer=restore_synthesizer,
        generator=generator,
        allow_target_leakage=CONFIG["allow_target_leakage"]
    )

    print(f"[CLIENT] Connecting to {CONFIG['server_address']}...")
    fl.client.start_numpy_client(server_address=CONFIG["server_address"], client=client)


if __name__ == "__main__":
    main()
