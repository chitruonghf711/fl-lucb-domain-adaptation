# Domain Adaptation of Federated Learning by Data Generation and Server Feedback

This project implements a federated learning (FL) system using [Flower](https://flower.dev/), where:

- The **server** coordinates global training, monitors target accuracy, and uses budgeted **LUCB** minibatches to select weak client models before generating SHAP feedback.
- The **clients** train local models on private data and, by default, use only weak labels plus aggregate SHAP feature ranks to guide local **TVAE** augmentation.

---

## Project Structure

```
.
├── client_tabular.py         # Federated client logic (PyTorch + SDV)
├── server_tabular.py         # Federated server logic (Flower + SHAP)
├── dataset/7030/
│   ├── test_server.csv       # Global test set for evaluation
│   ├── train1.csv            # Client 1 data
│   ├── train2.csv            # Client 2 data
│   └── ...                   # train3.csv, train4.csv, train5.csv
├── tvae_gen.pkl              # Pre-trained TVAE generator model
├── logs/                     # All run artefacts, one folder per run (log_dir)
│   └── lucb_pilot/
│       ├── server/           # summary_log.csv, summary_time.csv, server_config.json
│       │   ├── global_monitoring.jsonl # Full global-model monitoring, excluded from LUCB M
│       │   ├── audit/              # Sanitized LUCB selection audit per feedback event
│       │   └── feedback_logs/      # Abstract feedback per selected client
│       └── clients/
│           └── client1/      # log_<timestamp>.txt (one file per client run)
├── requirements.txt          # Combined dependencies (server + client)
└── README.md                 # Project documentation
```

---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/CaoDien2003/flgenai.git
cd flgenai/tabular
```

2. **Create and activate a conda environment:**
```bash
conda create -n flgenai-tabular python=3.10 -y
conda activate flgenai-tabular
```

> Requires Python 3.8–3.11 (verified on 3.10.20 via conda). Python ≥ 3.12 is **not** supported — `torch==2.2.2` has no wheel for it.

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## How to Run

> `server_address` defaults to `0.0.0.0:9000` on the server. Edit `CONFIG["server_address"]` in `server_tabular.py` if you need a different port.

### Start the Server

```bash
python server_tabular.py --selection-mode lucb --feedback-protocol abstract --log-dir logs/lucb_pilot
```

- Loads `dataset/7030/test_server.csv`
- Runs for N rounds (configurable)
- Evaluates the global model and independently logs plateau monitoring
- On a plateau, scores only planned LUCB minibatches of client models and sends feedback only for selected client/class errors

### Start Clients (each in separate terminal or machine)

Run once per client, each with its own `--client-id` and `--train-csv`:

```bash
python client_tabular.py --client-id client1 --train-csv dataset/7030/train1.csv --server-address 127.0.0.1:9000 --log-dir logs/lucb_pilot
python client_tabular.py --client-id client2 --train-csv dataset/7030/train2.csv --server-address 127.0.0.1:9000 --log-dir logs/lucb_pilot
# ...
```

Each client will:
- Load its own training CSV
- Train a local model
- (Optionally) receive abstract feedback from the server, keyed to its `--client-id`
- Use a local TVAE model to generate class-targeted synthetic rows and prefer sparse local regions of the ranked features

> `--client-id` must be unique per client — the server uses it to keep each client's feedback separate.
> `--log-dir` must match `CONFIG["log_dir"]` on the server so every log of one run lands under the same folder; it defaults to `logs/lucb_pilot`. Toggle `use_feedback` in `CONFIG` inside `client_tabular.py`.
> If two fit results report the same stable `--client-id`, the server withholds feedback for that event and writes a sanitized routing-guard audit instead of risking delivery to the wrong client.

---

## Features

- SHAP explainability to generate interpretable feedback
- TVAE-based synthetic data generation
- Cosine similarity filtering for high-quality augmentation
- Plateau detection for dynamic feedback
- Modular & easy-to-extend system (e.g. supports SDV, BERT, ResNet)

---

## LUCB Configuration

The server exposes the following pilot defaults through CLI flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--selection-mode` | `lucb` | `lucb` uses bounded minibatches; `legacy_full_scan` is an explicit compatibility baseline. |
| `--lucb-budget-fraction` | `0.25` | Fraction of full client-by-sample evaluation cost available to LUCB. |
| `--lucb-batch-size` | `64` | Best-effort stratified, non-replacement batch size. |
| `--lucb-top-k` | automatic max 3 | Number of weak client arms retained. |
| `--lucb-epsilon`, `--lucb-delta` | `0.02`, `0.05` | LUCB boundary tolerance and a family-wise Wilson error budget allocated across planned looks. |
| `--weak-class-accuracy` | `0.85` | A class is weak only if its Wilson upper bound is below this threshold. |
| `--seed` | `42` | Deterministic minibatch-plan seed. |
| `--lucb-prioritization` | `strict` | Resets Wilson evidence when client weights change. `decay_priority` reuses scheduling priority only. |
| `--feedback-protocol` | `abstract` | Safe default: labels plus aggregate SHAP feature ranks only. |

`M` in each audit record is the number of model-sample evaluations requested by LUCB and never exceeds its budget. Full global-model monitoring is recorded separately in `global_monitoring.jsonl` and is not included in `M`.

The test-only command below runs five deterministic synthetic clients without Flower, training, TVAE, SHAP, or a dataset. It checks the LUCB budget, deterministic selection, abstract payload validation, and audit sanitization:

```bash
python smoke_lucb.py
```

For the real lightweight Flower transport smoke, run this PowerShell helper from `tabular/` in the pinned project environment:

```powershell
.\run_smoke_lucb.ps1
```

It starts one server plus all five CSV clients for two rounds. The server uses `--test-force-feedback` plus smoke-only permissive LUCB thresholds (`--lucb-epsilon 1.0`, `--weak-class-accuracy 1.0`) so the first scheduled feedback event can be exercised; normal pilot defaults remain unchanged. Each client uses one local epoch and `--smoke-no-generation`, which skips TVAE setup/generation but still validates the abstract payload delivered in round two. The helper creates a timestamped run folder, checks `M <= budget`, requires an emitted abstract payload, and rejects raw-content fields in both the sanitized audit and every feedback JSON. It does not modify any `result_*` directory.

## Legacy Compatibility

The historical full scan and raw masked-value protocol remain available only when both settings are explicit. This mode can transmit target-domain feature values and prints a warning at server startup:

```bash
python server_tabular.py --selection-mode legacy_full_scan --feedback-protocol legacy --allow-raw-feedback
python client_tabular.py --feedback-protocol legacy --allow-raw-feedback
```

Audit JSON remains sanitized even when the runtime legacy payload is permitted. Do not use legacy mode for sensitive target-domain data.

## Other Configuration

- `--client-id`, `--train-csv`, `--server-address`, `--log-dir` are passed as CLI flags to `client_tabular.py` (see *How to Run*).
- Everything else is edited in the `CONFIG` dictionary in each script:
  - `server_tabular.py`: `log_dir`, training rounds, plateau window/threshold, SHAP threshold, top-k features, and all LUCB flags above
  - `client_tabular.py`: epochs, batch size, learning rate, generation multiplier, cosine similarity cutoff, `use_feedback` toggle
- Point `log_dir` at a new folder for each run you want to keep — re-running with the same one overwrites its CSVs.

---

## Logs & Output

Everything a run produces lives under the run's `log_dir` (`logs/lucb_pilot` by default):

| Path | Content |
| --- | --- |
| `logs/lucb_pilot/server/summary_log.csv` | One row per round (round 0 = initial parameters): accuracy, loss, precision, recall, f1, confusion matrix |
| `logs/lucb_pilot/server/summary_time.csv` | Per-round wall clock: round, aggregation and SHAP feedback time |
| `logs/lucb_pilot/server/server_config.json` | The server `CONFIG` of that run plus `total_time_sec` |
| `logs/lucb_pilot/server/global_monitoring.jsonl` | Global-model metrics and plateau inputs; deliberately excluded from LUCB budget accounting |
| `logs/lucb_pilot/server/audit/feedback_round_<N>.json` | Sanitized selection configuration, fingerprint, `M`, budget, Wilson bounds, coverage, stop reason, and explanation cost |
| `logs/lucb_pilot/server/feedback_logs/feedback_round_<N>_client_<id>.json` | Abstract weak labels and aggregate feature ranking for selected client/class errors |
| `logs/lucb_pilot/clients/<client-id>/log_<timestamp>.txt` | Client log: restore/generation counts, final train size, per-epoch loss |

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sdv.metadata.metadata'`** (when loading `tvae_gen.pkl`): your installed `sdv` version doesn't match the one `tvae_gen.pkl` was fitted with — SDV's internal module layout changes across releases, so `pickle.load()` needs an exact match. `requirements.txt` pins `sdv==1.22.1` (read from the pickle's embedded `_fitted_sdv_version`); make sure you actually installed that version (`pip show sdv`) rather than a stale one from a previous env.
- **`ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module`** (or `state is not a legacy MT19937 state`): `tvae_gen.pkl` also carries a random state pickled by numpy ≥ 2.0, while `torch==2.2.2` requires numpy < 2. `client_tabular.py` handles this inside `load_generator()`, which patches numpy's pickle reconstructors while the file is read and restores them afterwards — nothing to do, just don't replace that helper with a plain `pickle.load()`.
- **`FileNotFoundError` on `test_server.csv` / `trainX.csv`**: run the scripts from inside `tabular/` (paths in `CONFIG` are relative to that directory), and confirm the file exists under `dataset/7030/`.
- **Client killed with no traceback (`Killed` in the terminal)**: this is the Linux OOM killer, not a bug — check `dmesg | tail -30` for an `Out of memory: Killed process ... python` line. TVAE training is memory-hungry; on WSL2, raise the memory cap in `%UserProfile%\.wslconfig` (`[wsl2]` / `memory=6GB`, then `wsl --shutdown` from PowerShell) and close other processes before retrying.

---

## Example Use Cases

- Data privacy–preserving learning in banks
- Debugging misclassifications using SHAP feedback
- Exploring how synthetic data affects model performance

---

## Notes

- No raw client data is shared between clients and server
- Default feedback exchanges only model weights, weak labels, and aggregate feature ranks; it contains no target row, `reference`, or feature value
- The `legacy` protocol is intentionally separate, requires `--allow-raw-feedback`, and is not privacy-preserving
- SDV is used locally by clients to generate data

---

## Credits

- [Flower](https://flower.dev) - Federated learning framework
- [SHAP](https://github.com/slundberg/shap) - Model interpretability
- [SDV](https://docs.sdv.dev/sdv) - Synthetic data generation (TVAE)

---


## Contact

For inquiries or feedback, feel free to reach out:

- **Email**: nguyencaodien2003@gmail.com
- **GitHub**: [CaoDien2003](https://github.com/CaoDien2003)
