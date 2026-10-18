# CIFAR 10 Feedback Track

This directory contains the experimental image implementation for federated
learning with Grad-CAM-guided data generation. It is not runnable from a fresh
clone until the CIFAR subset folders and a compatible `cifar_diffusion.pth`
checkpoint are supplied.

## Entry points

Use `server.py` for the Flower server and `client_1.py` as the canonical client
runtime. The `client_2.py` through `client_9.py` files remain compatibility
entry points and accept the same generic `--cid` argument.

```powershell
python server.py --target <target-subset> --clients 9 --selection-mode lucb
python client_1.py --data <client-subset> --cid client1
```

## LUCB diagnostics

`--selection-mode lucb` is the default feedback-selection mode. It evaluates
only planned target minibatches, ranks client models by error rate, and emits a
sanitized `lucb_audit_round_<N>.json` file for each feedback event. Important
parameters are `--lucb-budget-fraction`, `--lucb-batch-size`, `--lucb-top-k`,
`--lucb-epsilon`, `--lucb-delta`, and `--weak-class-accuracy`.

`--selection-mode legacy_full_scan` preserves the earlier full-target
evaluation path for comparison. The global-model plateau monitor remains a
separate cost and is recorded independently from the client diagnostic budget
in `global_monitoring.jsonl`.

## Feedback privacy modes

The default `--feedback-protocol abstract` sends only class labels and Grad-CAM
patch coordinates. The client anchors those coordinates in a same-label local
image; if none is available, it falls back to class-conditional generation.
Abstract feedback requires `--mode refine`.

The legacy raw-patch payload is available only with both options below:

```powershell
--feedback-protocol legacy --allow-raw-feedback
```

Audit files and feedback logs deliberately omit target pixels in either mode.

## Validation

Use `--validate-config` to check protocol choices and target path availability
without starting a Flower server. Full image validation requires external data,
a diffusion checkpoint, and typically a GPU.

Create a clean Python 3.8-3.11 environment before installing this experimental
track. `requirements.txt` pins Flower 1.8.0, protobuf 4.25.3, NumPy 1.26.4,
Torch 2.2.2, and TorchVision 0.17.2 as one compatible runtime set; install a
matching PyTorch CUDA wheel when GPU support is required. Do not mix these pins
with an existing TensorFlow/Flower environment that upgrades NumPy or protobuf.
