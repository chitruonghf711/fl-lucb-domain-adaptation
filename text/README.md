# Text track: LUCB minibatch feedback

This track trains a DistilBERT news classifier with Flower. The server keeps the
target-domain CSV locally, selects weak client models with budgeted LUCB
minibatches, derives coarse Grad-CAM position feedback, and sends that feedback
to the selected client for local headline generation.

The default is a privacy-preserving **abstract** protocol. It sends only weak
class IDs and weighted `beginning` / `middle` / `end` positions. It never sends
server text, source tokens, token IDs, or a target-row reference to a client.

## Setup

Run commands from this directory so the documented relative paths resolve as
shown:

```powershell
cd text
pip install -r requirements.txt
```

`requirements.txt` pins NumPy below 2 because the pinned Flower/Torch stack is
not compatible with NumPy 2 binary extensions.

The supplied shards are:

- `data/test_server.csv`: server-only target-domain data
- `data/client0.csv` through `data/client4.csv`: local client shards

The default generator is `google/gemma-2b-it`. It is gated, so set a Hugging
Face token before starting a generation-enabled client:

```powershell
$env:HF_TOKEN = "your_huggingface_token"
```

The client validates this credential at startup. For a no-generation baseline,
pass `--generator disabled` instead; no generator model is loaded or downloaded.

## Run an abstract LUCB experiment

Start the server first. It creates the stable label mapping that every client
should use.

```powershell
python server_textdata.py `
  --server-address 0.0.0.0:8080 `
  --num-rounds 30 `
  --run-dir runs/study01 `
  --selection-mode lucb `
  --feedback-protocol abstract
```

Then start one client per terminal. Give every process a stable `--client-id`
and its own local shard:

```powershell
python client_textdata.py `
  --client-id client0 `
  --train-csv data/client0.csv `
  --label-map runs/study01/label_map.json `
  --run-dir runs/study01_clients `
  --server-address 127.0.0.1:8080 `
  --feedback-protocol abstract
```

Repeat with `client1` through `client4` and the matching CSV. The server
defaults to five required clients. For a controlled development run, use the
same lower `--min-clients` value on the server as the number of client
processes you start. LUCB selection and audit records use this reported stable
client ID; Flower's transient proxy ID is used only to route the next-round
payload. A duplicated stable ID causes that feedback event to be skipped rather
than routed ambiguously. The client also verifies the server label-map
fingerprint and label count before training, so a local fallback map cannot be
silently aggregated with a different classifier head.

The server detects local-client overfit or a global-target plateau in `auto`
mode. To make a controlled feedback test fire on every completed round, add
`--feedback-trigger always`. The global monitoring scan is recorded separately
and is not charged to the LUCB budget.

`--num-rounds` is a total-round target, including a resumed checkpoint; the
server runs only the remaining rounds when checkpoints are present.

## Selection configuration

The pilot defaults are designed to make the evaluation cost explicit:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--selection-mode` | `lucb` | Budgeted LUCB or `legacy_full_scan` baseline |
| `--lucb-budget-fraction` | `0.25` | Fraction of the full client-target scan available to LUCB |
| `--lucb-batch-size` | `64` | Deterministic, best-effort stratified minibatch size |
| `--lucb-top-k` | `0` | `0` selects automatically, with a maximum of three clients |
| `--lucb-epsilon` | `0.02` | LUCB boundary tolerance |
| `--lucb-delta` | `0.05` | Family-wise Wilson error budget allocated across planned looks |
| `--weak-class-accuracy` | `0.85` | A class is weak only when its Wilson upper bound is below this value |
| `--seed` | `42` | Deterministic client-specific minibatch plans |
| `--lucb-prioritization` | `strict` | Reset evidence after weights change; `decay_priority` only reuses scheduling priority |

`M` is the model-sample evaluation count reported in every feedback audit. The
selector never evaluates rows outside a planned minibatch, never reuses a row
within one client plan, and never exceeds its selection budget. In strict mode,
new post-fit weights discard old Wilson evidence. The experimental
`decay_priority` mode retains only priority scheduling state, not old evidence.

Use the old complete scan only as an experimental baseline:

```powershell
python server_textdata.py `
  --run-dir runs/legacy_baseline `
  --selection-mode legacy_full_scan `
  --feedback-protocol abstract
```

## Feedback protocols

### Abstract (default)

An abstract text payload has this shape:

```json
{
  "protocol": "abstract",
  "domain": "text",
  "weak_labels": [3, 17],
  "salient_positions": [
    {"position": "beginning", "weight": 0.61},
    {"position": "middle", "weight": 0.39}
  ]
}
```

The client generates a prompt from the label and position distribution. It does
not perform token-match filtering because it receives no server tokens.

### Legacy token feedback (explicit opt-in)

Legacy mode is for baseline comparison only. Both endpoints must use both
flags; startup emits a warning:

```powershell
python server_textdata.py --feedback-protocol legacy --allow-raw-feedback
python client_textdata.py --feedback-protocol legacy --allow-raw-feedback
```

Legacy runtime payloads may contain token strings. They are never copied into
server audit JSON or client feedback-receipt audit JSON. Do not enable this mode
with sensitive target-domain data.

## Outputs and audit trail

Server output under `--run-dir` contains:

- `label_map.json`: stable global category-to-ID mapping
- `checkpoints/round_*.pt`: aggregate-model checkpoints
- `global_monitoring.jsonl`: one metadata-only global-monitoring record per
  round; its model-sample count is explicitly excluded from LUCB `M`
- `feedback_status.jsonl`: explicit skips caused by missing client results or
  ambiguous stable client identifiers
- `feedback/round_*.json`: outgoing runtime feedback, which contains no raw
  content in abstract mode
- `audit/lucb_audit_feedback_round_*.json`: sanitized event audit with configuration,
  model fingerprint, `M`, explanation cost, selected clients/classes, Wilson
  bounds, minibatch coverage, boundary gap, stop reason, and separate global
  monitoring cost

Each client writes under `<run-dir>/<client-id>/`:

- `checkpoints/round_*.pt`: local model checkpoints
- `generated/feedback/round_*/class_*/headlines.json`: locally generated rows
- `audit/feedback_receipt_round_*.json`: metadata-only receipt

`--save-merged-data` additionally writes a local real-plus-synthetic CSV. It is
off by default because it can be large and contains local training data.

## Validation and limitations

Run these dependency-light checks from the repository root:

```powershell
python -m py_compile text/server_textdata.py text/client_textdata.py
python -m unittest discover -s tests -p "test_*.py" -v
python text/server_textdata.py --help
python text/client_textdata.py --help
```

The repository test suite validates the dependency-free LUCB planner, Wilson
intervals, budget enforcement, deterministic plans, exhausted arms, strict
reset behavior, decay-priority isolation, audit sanitization, and both feedback
protocols. It does not run a full text training session: that requires the
DistilBERT assets, five client processes, and, for Gemma generation, a permitted
Hugging Face account and adequate GPU memory.

Grad-CAM is intentionally calculated only for errors already evaluated by the
selected client model. In abstract mode, it is reduced to a position
distribution before transport. Historical `result_text` artifacts are not read,
modified, or used as reproduction evidence.
