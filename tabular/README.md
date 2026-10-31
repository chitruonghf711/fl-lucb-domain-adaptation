# Federated Domain Adaptation with Minibatch Cumulative Evaluation

This project implements a federated learning (FL) system using [Flower](https://flower.dev/) for domain adaptation across tabular churn datasets.

---

## Key Highlights & Innovations in this Branch (`minibatch`)

### 1. Minibatch Cumulative Accuracy Evaluation & Early Stopping
In standard Federated Learning with domain adaptation, evaluating every client model across a large centralized target test set ($D_T$) incurs massive computation and latency at the server.

In this branch:
- The server partitions the test set into minibatches (stratified or sequential, size configured by `--minibatch-eval-size 300`).
- Models are evaluated batch-by-batch up to `--max-eval-minibatches 10`.
- Cumulative accuracy is computed dynamically:
  $$\text{Acc}_{\text{cumulative}} = \frac{\sum \text{Correct}}{\sum \text{Tested}}$$
- **Early Stopping**: If cumulative accuracy reaches `--client-good-threshold-acc` (e.g. `0.85`), the client is classified as **`GOOD`**, and evaluation halts immediately. This saves up to 70%–90% of forward-pass computations at the server (`samples_saved_pct`).
- If accuracy remains below threshold after all minibatches, the client is classified as **`NEEDS_FEEDBACK`**, and the server prepares adaptation feedback.
- Full evaluation trajectories are recorded in `minibatch_evaluation_log.csv`.

---

### 2. Privacy-Preserving Feedback & Test Data Leakage Prevention
In earlier iterations of baseline code, server feedback transmitted raw feature values of misclassified target-domain test samples, which clients then restored and concatenated directly into their local training sets (`restored_df`). This constituted severe **Test Data Leakage**, artificially inflating test accuracy at the expense of privacy and scientific validity.

This branch resolves this issue completely:
1. **Abstract Feedback Protocol (`--feedback-protocol abstract`, default)**:
   - The server identifies the weak class label and the top-$k$ most important features (via SHAP or gradient importance).
   - **ZERO raw or masked test samples** leave the server.
   - The client uses its local TVAE generator to synthesize samples conditioned on the weak class without any target data leakage.
2. **Safe Detailed Feedback Protocol (`--feedback-protocol detailed`)**:
   - If legacy masked SHAP feedback is used, restored samples are used **strictly for guiding synthetic data filtering** (`collect_synthetic_samples`).
   - By default (`--allow-target-leakage False`), restored test samples are **never** concatenated into the client's training dataset.
   - The `--allow-target-leakage` flag is retained only for controlled ablation experiments reproducing legacy leakage behavior.
3. **Round-based Feedback Storage Reset**:
   - `feedback_storage` is refreshed every round, ensuring stale feedback is not accumulated indefinitely.

---

## Project Structure

```
.
├── client_tabular.py         # Federated client logic (PyTorch + SDV + Data Leakage Guard)
├── server_tabular.py         # Federated server logic (Flower + Minibatch Eval + SHAP)
├── smoke_minibatch.py        # Fast end-to-end smoke test script
├── dataset/7030/
│   ├── test_server.csv       # Global test set for evaluation (3,000 samples)
│   ├── train1.csv            # Client 1 data (1,050 samples)
│   ├── train2.csv            # Client 2 data (2,100 samples)
│   └── ...                   # train3.csv, train4.csv, train5.csv
├── tvae_gen.pkl              # Pre-trained TVAE generator model
├── logs/                     # Experiment output folder
│   └── exp_minibatch/
│       ├── server/
│       │   ├── summary_log.csv               # Global round metrics
│       │   ├── summary_time.csv              # Wall-clock timings
│       │   ├── minibatch_evaluation_log.csv  # Minibatch evaluation & early stopping log
│       │   ├── server_config.json            # Run hyperparameters & metadata
│       │   └── feedback_logs/                # Individual feedback JSON files
│       └── clients/
│           └── client1/                      # Client local training logs
└── README.md
```

---

## How to Run

### 1. Run the Smoke Test
Verify minibatch evaluation, early stopping, sample savings, abstract feedback, and data leakage safeguards:
```bash
python tabular/smoke_minibatch.py
```

### 2. Start the Server with Minibatch Evaluation
```bash
python tabular/server_tabular.py \
    --server-address 0.0.0.0:9000 \
    --minibatch-eval-size 300 \
    --max-eval-minibatches 10 \
    --client-good-threshold-acc 0.85 \
    --feedback-protocol abstract \
    --log-dir logs/exp_minibatch
```

### 3. Start Clients (in separate terminals)
```bash
python tabular/client_tabular.py \
    --client-id client1 \
    --train-csv tabular/dataset/7030/train1.csv \
    --server-address 127.0.0.1:9000 \
    --log-dir logs/exp_minibatch

python tabular/client_tabular.py \
    --client-id client2 \
    --train-csv tabular/dataset/7030/train2.csv \
    --server-address 127.0.0.1:9000 \
    --log-dir logs/exp_minibatch
```

*(Repeat for clients 3, 4, 5 with their respective training CSV files).*

---

## Comparing Minibatch Thresholding vs. LUCB (Fair Benchmark)

| Dimension | Minibatch Thresholding (`minibatch` branch) | Minibatch LUCB (`tan` branch) | Full-Scan Baseline (Paper) |
| :--- | :--- | :--- | :--- |
| **Server Evaluation Cost** | Sequential early stopping per client | Multi-armed bandit budget ($M \le E$) | Full batch ($100\%$ samples) |
| **Selection Criterion** | Accuracy threshold ($\ge \theta$) | Top-$k$ lowest accuracy via upper confidence bounds | All clients on plateau |
| **Data Privacy** | Abstract feedback (Zero test leakage) | Abstract feedback (Zero test leakage) | Masked test samples (Historical leakage fixed) |
| **Statistical Guarantee** | Heuristic early stopping | $(\epsilon, \delta)$-PAC bandit bounds | Exhaustive empirical mean |

---

## Artifacts & Logs

After running, check `logs/exp_minibatch/server/minibatch_evaluation_log.csv`:
```csv
round,client_id,batches_tested,samples_tested,total_available_samples,samples_saved_pct,cumulative_accuracy,threshold,status,eval_time_sec,shap_time_sec,feedback_samples_count
2,client1,10,3000,3000,0.0,0.7966,0.85,NEEDS_FEEDBACK,0.01,1.25,610
6,client4,3,900,3000,70.0,0.8544,0.85,GOOD,0.0,0.0,0
```
- `samples_saved_pct` measures the percentage of test evaluations saved by early stopping.
- `feedback_samples_count` tracks adaptation signals dispatched to the client.
