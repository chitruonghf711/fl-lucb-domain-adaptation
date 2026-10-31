#!/usr/bin/env python3
"""
Run:
    python server.py --target /data/subset1 --clients 9 --method proposed
"""

import argparse
import hashlib
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fl_lucb import (BatchScore, LUCBSelector, MinibatchPlan, SelectionConfig,
                     SelectionState, build_audit_record, validate_feedback_payload,
                     write_audit_json)

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from flwr.common import (FitIns, Parameters, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader


IMG_SIZE = 32
NUM_CLASSES = 10


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# Classifier: ResNet-18  

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # changed 16 -> 64
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    return ResNet(ResNetBlock, [2, 2, 2, 2], num_classes=num_classes)


# Data
class CifarSubsetFolder(ImageFolder):
    """ImageFolder that forces class index = the number i in folder name `class_{i}_{name}`."""

    def find_classes(self, directory):
        classes = sorted(e.name for e in os.scandir(directory) if e.is_dir())
        class_to_idx = {c: int(c.split("_")[1]) for c in classes}
        return classes, class_to_idx


def load_subset(root: str, verbose: bool = True):
    """Load one non-IID subset. ToTensor -> [0,1]"""
    from collections import Counter
    ds = CifarSubsetFolder(root, transform=transforms.Compose([transforms.ToTensor()]))
    if verbose:
        print(f"[data] {root}: {len(ds)} images | class_to_idx = {ds.class_to_idx}")
        print(f"[data] distribution: {dict(sorted(Counter(ds.targets).items()))}")
    return ds

NUM_ROUNDS = 100        # number of communication rounds
N_CLIENTS_FB = 3       # "the three with the lowest accuracy"
N_CLASSES_FB = 3       # "a few classes with the highest error rates"
N_PER_CLASS = 200      # 200 images/class -> 3 x 200 = 600 images/client
TOP_K = 8              # number of Grad-CAM cells sent
TOP_M = 20             # number of misclassified samples, highest confidence
GRID = 8               # 8x8 CAM grid -> 4x4 px patches
GEN_MODE = "refine"
METHODS = ("proposed", "nofeedback", "fedavg", "fedcog")

# FedCOG (github.com/rui-ye/FedCOG)
FEDCOG_START = 50      # --start_round: plain FedAvg before this round
FEDCOG_N = 256         # --synthesis_batch_size
FEDCOG_STEPS = 500     # --g_steps
FEDCOG_LR = 0.01       # --lr_g
FEDCOG_ADV = 0.1       # --adv
FEDCOG_T = 1.0         # --T / --js_T

DEVICE = get_device()
SERVER_LOG_DIR = os.environ.get("FLGENAI_LOG_DIR", "./server_logs")

SERVER_HOST = "0.0.0.0"   
SERVER_PORT = 8081
SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"


def grad_cam(model: nn.Module, x: torch.Tensor, y: int, target_layer: nn.Module) -> torch.Tensor:
    """Grad-CAM for one image. x: (3,H,W). Returns CAM (Hf,Wf) normalized to [0,1]."""
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x.unsqueeze(0))
        logits[0, y].backward()
        a, g = acts["v"][0], grads["v"][0]
        w = g.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((w * a).sum(0)).detach()
    finally:
        h1.remove(); h2.remove()
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def build_feedback_entry(x01: torch.Tensor, y: int, cam: torch.Tensor, sample_id: int,
                         client_prob: float, top_k: int = TOP_K,
                         grid: int = GRID, feedback_protocol: str = "legacy") -> Dict[str, Any]:
    patch = IMG_SIZE // grid
    cam_g = F.interpolate(cam[None, None], size=(grid, grid),
                          mode="bilinear", align_corners=False)[0, 0]
    idx = cam_g.flatten().topk(min(top_k, grid * grid)).indices.tolist()

    positions, salient_patches, values = [], [], []
    for i in idx:
        r, c = divmod(i, grid)
        positions.append([r, c])
        salient_patches.append({
            "row": int(r),
            "column": int(c),
            "weight": float(cam_g[r, c].item()),
        })
        if feedback_protocol == "legacy":
            values.append(x01[:, r * patch:(r + 1) * patch,
                                 c * patch:(c + 1) * patch].detach().cpu().numpy().tolist())
    if feedback_protocol == "abstract":
        return validate_feedback_payload(
            {
                "protocol": "abstract",
                "domain": "image",
                "weak_labels": [int(y)],
                "generation_mode": "refine",
                "patch_grid_size": int(grid),
                "salient_patches": salient_patches,
            },
            protocol="abstract",
            domain="image",
            allow_raw_feedback=False,
        )
    entry = {
        "protocol": "legacy",
        "domain": "image",
        "weak_labels": [int(y)],
        "schema_version": 2,
        "label": int(y),
        "grid": grid,
        "patch": patch,
        "positions": positions,
        "client_prob": float(client_prob),
    }
    if feedback_protocol != "legacy":
        raise ValueError(f"Unsupported feedback protocol: {feedback_protocol}")
    entry["sample_id"] = int(sample_id)
    entry["values"] = values
    return validate_feedback_payload(
        entry,
        protocol="legacy",
        domain="image",
        allow_raw_feedback=True,
    )


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device=None) -> torch.Tensor:
    """prob(W_j(D^t)) — Algorithm 1 line 4."""
    device = device or DEVICE
    model.eval()
    out = [torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in loader]
    return torch.cat(out)


def select_trustworthy(probs: Dict[str, torch.Tensor], labels: torch.Tensor):

    keys = list(probs)
    stack = torch.stack([probs[k] for k in keys])                    # (M, N, C)
    correct = stack.argmax(2) == labels[None, :]                     # (M, N)
    conf_y = stack.gather(2, labels.view(1, -1, 1).expand(len(keys), -1, 1)).squeeze(2)
    conf_y = conf_y.masked_fill(~correct, -1.0)
    best = conf_y.argmax(0)
    return [keys[i] for i in best.tolist()], correct.any(0)


def per_class_error(pred: torch.Tensor, labels: torch.Tensor) -> Dict[int, float]:
    err = {}
    for c in range(NUM_CLASSES):
        m = labels == c
        if int(m.sum()) > 0:
            err[c] = float((pred[m] != c).float().mean())
    return err


def algorithm1(global_model: nn.Module, local_models: Dict[str, nn.Module],
               test_set, test_loader: DataLoader, target_layer_of,
               chosen_clients: List[str], n_classes: int = N_CLASSES_FB,
               top_k: int = TOP_K, top_m: int = TOP_M, grid: int = GRID,
               device=None, names: Dict[str, str] = None,
               feedback_protocol: str = "legacy") -> Dict[str, List[Dict[str, Any]]]:

    device = device or DEVICE
    names = names or {}
    labels = torch.tensor([int(test_set[i][1]) for i in range(len(test_set))])

    probs = {"__global__": predict_probs(global_model, test_loader, device)}     # lines 3-5
    for cid, m in local_models.items():
        probs[cid] = predict_probs(m, test_loader, device)

    best_of, has_correct = select_trustworthy(probs, labels)                     # line 9
    models = {"__global__": global_model, **local_models}

    feedback: Dict[str, List[Dict[str, Any]]] = {}
    for cid in chosen_clients:
        p = probs[cid]
        pred, conf = p.argmax(1), p.max(1).values
        wrong = pred != labels                                                   # line 13

        eligible = wrong & has_correct
        avail = set(labels[eligible].tolist())
        err = per_class_error(pred, labels)
        top_cls = sorted([c for c in err if c in avail],
                         key=err.get, reverse=True)[:n_classes]
        if len(top_cls) < n_classes:
            print(f"    [{names.get(cid, cid)}] only {len(top_cls)}/{n_classes} classes feasible "
                  f"(no model ever predicted the other classes correctly)")

        quota = max(1, top_m // max(len(top_cls), 1))
        cand, spare = [], []
        for c in top_cls:
            pool = [(i, float(conf[i])) for i in range(len(labels))              
                    if wrong[i] and int(labels[i]) == c and has_correct[i]]
            pool.sort(key=lambda t: -t[1])                                      
            cand.extend(pool[:quota])
            spare.extend(pool[quota:])
        spare.sort(key=lambda t: -t[1])
        cand.extend(spare[:max(0, top_m - len(cand))])                          
        cand.sort(key=lambda t: -t[1])

        entries = []
        for sid, cf in cand:
            model = models[best_of[sid]]
            x, y = test_set[sid]
            cam = grad_cam(model, x.to(device), int(y), target_layer_of(model))  
            entries.append(build_feedback_entry(
                x, int(y), cam, sid, cf, top_k, grid, feedback_protocol=feedback_protocol
            ))
        feedback[cid] = entries                                                 

        covered = sorted({
            int(label)
            for entry in entries
            for label in entry.get("weak_labels", [entry.get("label")])
            if label is not None
        })
        print(f"    [{names.get(cid, cid)}] worst classes {top_cls} "
              f"(err {[round(err[c], 3) for c in top_cls]}) -> {len(entries)} samples "
              f"covering {len(covered)} classes {covered}")
    return feedback


class FeedbackFedAvg(FedAvg):

    def __init__(self, test_set, target_layer_of, *,
                 min_rounds: int = 5,
                 acc_drop_tol: float = 0.0,
                 plateau_patience: int = 3, plateau_delta: float = 2e-3,
                 overfit_gap: float = 0.15,
                 n_clients: int = N_CLIENTS_FB, n_classes: int = N_CLASSES_FB,
                 n_per_class: int = N_PER_CLASS, diff_steps: int = 500,
                 top_k: int = TOP_K,
                 top_m: int = TOP_M, grid: int = GRID, gen_mode: str = GEN_MODE,
                 use_feedback: bool = True, method: str = "proposed",
                 batch_size: int = 128, selection_mode: str = "lucb",
                 lucb_budget_fraction: float = 0.25, lucb_batch_size: int = 64,
                 lucb_top_k: Optional[int] = None, lucb_epsilon: float = 0.02,
                 lucb_delta: float = 0.05, weak_class_accuracy: float = 0.85,
                 lucb_prioritization: str = "strict", seed: int = 42,
                 feedback_protocol: str = "abstract", allow_raw_feedback: bool = False,
                 force_feedback: bool = False, allow_partial_feedback: bool = False, device=None, **kwargs):
        super().__init__(**kwargs)
        self.test_set     = test_set
        self.test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        self.target_layer_of = target_layer_of
        self.min_rounds   = min_rounds
        self.acc_drop_tol = acc_drop_tol
        self.plateau_patience = plateau_patience
        self.plateau_delta    = plateau_delta
        self.overfit_gap  = overfit_gap
        self.n_clients    = n_clients
        self.n_classes    = n_classes
        self.n_per_class  = n_per_class
        self.diff_steps   = diff_steps
        self.top_k, self.top_m, self.grid = top_k, top_m, grid
        self.gen_mode     = gen_mode
        self.use_feedback = use_feedback   
        self.method       = method        
        self.device       = device or DEVICE
        self.selection_mode = selection_mode
        self.feedback_protocol = feedback_protocol
        self.allow_raw_feedback = allow_raw_feedback
        self.force_feedback = force_feedback
        self.lucb_config = SelectionConfig(
            selection_mode=selection_mode,
            lucb_budget_fraction=lucb_budget_fraction,
            lucb_batch_size=lucb_batch_size,
            lucb_top_k=None if lucb_top_k is None or lucb_top_k <= 0 else lucb_top_k,
            lucb_epsilon=lucb_epsilon,
            lucb_delta=lucb_delta,
            weak_class_accuracy=weak_class_accuracy,
            seed=seed,
            lucb_prioritization=lucb_prioritization,
            allow_partial_feedback=allow_partial_feedback,
        )
        self.lucb_selector = LUCBSelector(self.lucb_config)
        self.lucb_state = SelectionState()
        self._latest_observations: Dict[str, Dict[int, Dict[str, Any]]] = {}

        self.acc_hist: List[float] = []          # global accuracy per round
        self.local_acc: Dict[str, float] = {}    # each client's accuracy on the server test set
        self.local_acc_hist: List[Dict[str, float]] = []   # local_acc per round, by --cid
        self.names: Dict[str, str] = {}          # Flower proxy.cid -> client --cid
        self.pending: Dict[str, List[Dict[str, Any]]] = {}
        self.fb_rounds: List[int] = []           # rounds that actually produced feedback

    def _to_model(self, nds: List[np.ndarray]) -> nn.Module:
        m = resnet18(NUM_CLASSES)
        m.load_state_dict({k: torch.tensor(v)
                           for k, v in zip(m.state_dict().keys(), nds)}, strict=True)
        return m.to(self.device).eval()

    @torch.no_grad()
    def _accuracy(self, model: nn.Module) -> float:
        correct = seen = 0
        model.eval()
        for x, y in self.test_loader:
            pred = model(x.to(self.device)).argmax(1).cpu()
            correct += int((pred == y).sum()); seen += len(y)
        return correct / max(seen, 1)

    def _model_fingerprint(self, local_models: Dict[str, nn.Module]) -> str:
        digest = hashlib.sha256()
        for client_id in sorted(local_models):
            digest.update(client_id.encode("utf-8"))
            for tensor in local_models[client_id].state_dict().values():
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _select_lucb(self, local_models: Dict[str, nn.Module], server_round: int):
        labels = [int(self.test_set[index][1]) for index in range(len(self.test_set))]
        sample_ids = list(range(len(self.test_set)))
        plans = {
            client_id: MinibatchPlan.build(
                client_id=client_id,
                labels=labels,
                sample_ids=sample_ids,
                batch_size=self.lucb_config.lucb_batch_size,
                seed=self.lucb_config.seed,
            )
            for client_id in local_models
        }
        self._latest_observations = {client_id: {} for client_id in local_models}

        def scorer(client_id, batch):
            indices = [int(sample_id) for sample_id in batch.sample_ids]
            inputs = torch.stack([self.test_set[index][0] for index in indices]).to(self.device)
            with torch.no_grad():
                probabilities = torch.softmax(local_models[client_id](inputs), dim=1).cpu()
            predictions = probabilities.argmax(1).tolist()
            confidences = probabilities.max(1).values.tolist()
            for sample_id, label, prediction, confidence in zip(
                indices, batch.labels, predictions, confidences
            ):
                self._latest_observations[client_id][sample_id] = {
                    "label": int(label),
                    "prediction": int(prediction),
                    "confidence": float(confidence),
                }
            return BatchScore.from_predictions(batch.labels, predictions)

        return self.lucb_selector.select(
            plans,
            scorer,
            model_fingerprint=self._model_fingerprint(local_models),
            state=self.lucb_state,
            event_id=f"image-round-{server_round}",
        )

    def _build_lucb_feedback(self, local_models: Dict[str, nn.Module], selection) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
        if selection.status != "ok" or not selection.feedback_ready:
            return {}, 0
        feedback: Dict[str, List[Dict[str, Any]]] = {}
        explanation_evaluations = 0
        for client_id in selection.selected_client_ids:
            weak_classes = {
                int(item.label)
                for item in selection.selected_classes.get(client_id, ())
                if item.status == "weak"
            }
            observations = self._latest_observations.get(client_id, {})
            candidates = [
                (sample_id, observation)
                for sample_id, observation in observations.items()
                if observation["prediction"] != observation["label"]
                and observation["label"] in weak_classes
            ]
            if not candidates:
                feedback[client_id] = []
                continue
            grouped: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
            for sample_id, observation in candidates:
                grouped.setdefault(observation["label"], []).append((sample_id, observation))
            quota = max(1, self.top_m // max(len(grouped), 1))
            selected: List[Tuple[int, Dict[str, Any]]] = []
            spare: List[Tuple[int, Dict[str, Any]]] = []
            for label in sorted(grouped):
                ordered = sorted(grouped[label], key=lambda item: -item[1]["confidence"])
                selected.extend(ordered[:quota])
                spare.extend(ordered[quota:])
            selected.extend(sorted(spare, key=lambda item: -item[1]["confidence"])[:max(0, self.top_m - len(selected))])
            entries = []
            for sample_id, observation in selected[:self.top_m]:
                image, label = self.test_set[sample_id]
                cam = grad_cam(
                    local_models[client_id], image.to(self.device), int(label),
                    self.target_layer_of(local_models[client_id]),
                )
                entries.append(build_feedback_entry(
                    image, int(label), cam, sample_id, observation["confidence"], self.top_k,
                    self.grid, feedback_protocol=self.feedback_protocol,
                ))
                explanation_evaluations += 1
            feedback[client_id] = entries
        return feedback, explanation_evaluations

    def _audit_configuration(self) -> Dict[str, Any]:
        """Return effective safe configuration values for one feedback audit."""

        configuration = self.lucb_config.to_dict()
        configuration.update({
            "selection_mode": self.selection_mode,
            "feedback_protocol": self.feedback_protocol,
            "legacy_opt_in": bool(self.allow_raw_feedback),
            "generation_mode": self.gen_mode,
            "selected_client_cap": self.n_clients,
            "selected_class_cap": self.n_classes,
            "top_m": self.top_m,
            "top_k": self.top_k,
            "patch_grid_size": self.grid,
        })
        return configuration

    @staticmethod
    def _lucb_delivery_state(selection) -> Dict[str, Any]:
        """Describe whether a completed LUCB result may produce feedback."""

        if selection.status != "ok":
            return {
                "status": "not_delivered",
                "reason": "selection_status_not_ok",
                "selection_status": selection.status,
                "selection_stop_reason": selection.stop_reason,
            }
        if not selection.feedback_ready:
            return {
                "status": "not_delivered",
                "reason": "no_confident_weak_class",
                "selection_status": selection.status,
                "selection_stop_reason": selection.stop_reason,
            }
        return {
            "status": "eligible",
            "reason": None,
            "selection_status": selection.status,
            "selection_stop_reason": selection.stop_reason,
        }

    def _write_lucb_audit(
        self,
        server_round: int,
        selection,
        explanation_evaluations: int,
        delivery: Dict[str, Any] = None,
    ) -> None:
        if delivery is None:
            delivery = self._lucb_delivery_state(selection)
        configuration = dict(selection.audit.get("configuration", {}))
        configuration.update(self._audit_configuration())
        audit = build_audit_record(
            event_id=f"image-round-{server_round}",
            configuration=configuration,
            model_fingerprint=selection.model_fingerprint,
            status=selection.status,
            stop_reason=selection.stop_reason,
            model_sample_evaluations=selection.model_sample_evaluations,
            budget=selection.budget,
            selected_client_ids=selection.selected_client_ids,
            selected_classes=selection.selected_classes,
            confidence_bounds=selection.confidence_bounds,
            coverage=selection.coverage,
            scored_batches=selection.scored_batches,
            boundary_gap=selection.boundary_gap,
            explanation_cost={
                "grad_cam_evaluations": int(explanation_evaluations),
                "unit": "selected_model_sample_explanations",
            },
            protocol=self.feedback_protocol,
            domain="image",
            feedback_protocol=self.feedback_protocol,
            metadata={
                "round": server_round,
                "feedback_delivery": delivery,
                "global_monitoring_evaluations_excluded_from_lucb": len(self.test_set),
            },
        )
        write_audit_json(
            os.path.join(SERVER_LOG_DIR, f"lucb_audit_round_{server_round}.json"), audit
        )

    def _write_legacy_full_scan_audit(
        self,
        server_round: int,
        local_models: Dict[str, nn.Module],
        chosen_clients: List[str],
        feedback: Dict[str, List[Dict[str, Any]]],
        trigger_reason: str,
    ) -> None:
        """Write a sanitized audit for the explicit full-scan compatibility path."""

        sample_count = len(self.test_set)
        local_accuracy_evaluations = len(local_models) * sample_count
        trustworthy_model_evaluations = (len(local_models) + 1) * sample_count
        model_sample_evaluations = local_accuracy_evaluations + trustworthy_model_evaluations
        delivered_entries = sum(len(entries) for entries in feedback.values())
        delivery = {
            "status": "delivered" if delivered_entries else "not_delivered",
            "reason": None if delivered_entries else "no_eligible_legacy_errors",
            "entry_count": delivered_entries,
        }
        coverage = {
            client_id: {
                "scored_sample_count": sample_count,
                "mode": "legacy_full_scan",
            }
            for client_id in sorted(local_models)
        }
        audit = build_audit_record(
            event_id=f"image-round-{server_round}",
            configuration=self._audit_configuration(),
            model_fingerprint=self._model_fingerprint(local_models),
            status="ok",
            stop_reason="legacy_full_scan_complete",
            model_sample_evaluations=model_sample_evaluations,
            budget=model_sample_evaluations,
            selected_client_ids=chosen_clients,
            selected_classes={client_id: () for client_id in chosen_clients},
            confidence_bounds={},
            coverage=coverage,
            scored_batches={},
            boundary_gap=None,
            explanation_cost={
                "grad_cam_evaluations": delivered_entries,
                "unit": "selected_model_sample_explanations",
            },
            protocol=self.feedback_protocol,
            domain="image",
            feedback_protocol=self.feedback_protocol,
            metadata={
                "round": server_round,
                "feedback_delivery": delivery,
                "trigger_reason": trigger_reason,
                "local_accuracy_model_sample_evaluations": local_accuracy_evaluations,
                "trustworthy_model_sample_evaluations": trustworthy_model_evaluations,
                "global_monitoring_evaluations_excluded_from_lucb": sample_count,
            },
        )
        write_audit_json(
            os.path.join(SERVER_LOG_DIR, f"legacy_full_scan_audit_round_{server_round}.json"), audit
        )

    def _write_global_monitoring(self, server_round: int, accuracy: float) -> None:
        """Log full global-model monitoring outside the LUCB diagnostic budget."""

        os.makedirs(SERVER_LOG_DIR, exist_ok=True)
        record = {
            "round": server_round,
            "global_accuracy": float(accuracy),
            "model_sample_evaluations": len(self.test_set),
            "included_in_lucb_budget": False,
        }
        path = os.path.join(SERVER_LOG_DIR, "global_monitoring.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    def _should_trigger(self, server_round: int, results) -> Tuple[bool, str]:
        if self.force_feedback:
            return True, "forced feedback for a controlled experiment"
        if server_round < self.min_rounds:                                 
            return False, f"fewer than {self.min_rounds} rounds"

        h = self.acc_hist
        if len(h) >= 2 and h[-1] < h[-2] - self.acc_drop_tol:               
            return True, f"global dropped {h[-2]:.4f} -> {h[-1]:.4f}"

        if len(h) >= self.plateau_patience + 1:                              
            recent = h[-(self.plateau_patience + 1):]
            span = max(recent) - min(recent)
            if span < self.plateau_delta:
                return True, f"global plateaued for {self.plateau_patience} rounds (span {span:.4f})"

        if self.selection_mode == "legacy_full_scan":
            gaps = {}
            for proxy, fit_res in results:
                tr = fit_res.metrics.get("train_acc")
                if tr is not None and proxy.cid in self.local_acc:
                    gaps[proxy.cid] = float(tr) - self.local_acc[proxy.cid]
            if gaps and max(gaps.values()) > self.overfit_gap:
                worst = max(gaps, key=gaps.get)
                return True, f"overfit: {self.names.get(worst, worst)} train/test gap {gaps[worst]:.3f}"

        return False, "not needed"

    def aggregate_fit(self, server_round: int, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is None:
            return params, metrics

        global_model = self._to_model(parameters_to_ndarrays(params))
        local_models = {proxy.cid: self._to_model(parameters_to_ndarrays(fr.parameters))
                        for proxy, fr in results}
        self.names.update({proxy.cid: str(fr.metrics.get("client_id", proxy.cid))
                           for proxy, fr in results})
        name = lambda cid: self.names.get(cid, cid)

        acc_g = self._accuracy(global_model)
        self.acc_hist.append(acc_g)
        self._write_global_monitoring(server_round, acc_g)
        if self.selection_mode == "legacy_full_scan":
            self.local_acc = {cid: self._accuracy(model) for cid, model in local_models.items()}
            self.local_acc_hist.append({name(cid): accuracy for cid, accuracy in self.local_acc.items()})
            print(f"[R{server_round}] global acc {acc_g:.4f} | "
                  f"client { {name(cid): round(accuracy, 3) for cid, accuracy in sorted(self.local_acc.items(), key=lambda item: name(item[0]))} }")
        else:
            print(f"[R{server_round}] global acc {acc_g:.4f} | client diagnostics are budgeted")

        if self.method in ("fedavg", "fedcog"):
            return params, metrics            # baselines: no server-side generation

        trigger, why = self._should_trigger(server_round, results)            
        if not trigger:
            print(f"[R{server_round}] no feedback generated ({why})")
            return params, metrics

        if self.use_feedback:
            if self.selection_mode == "legacy_full_scan":
                chosen = sorted(self.local_acc, key=self.local_acc.get)[:self.n_clients]
                print(f"[R{server_round}] GENERATING LEGACY FEEDBACK ({why}) -> {len(chosen)} lowest "
                      f"clients: {[name(client_id) for client_id in chosen]}")
                self.pending = algorithm1(
                    global_model, local_models, self.test_set, self.test_loader,
                    self.target_layer_of, chosen, n_classes=self.n_classes, top_k=self.top_k,
                    top_m=self.top_m, grid=self.grid, device=self.device, names=self.names,
                    feedback_protocol=self.feedback_protocol,
                )
                self._write_legacy_full_scan_audit(
                    server_round, local_models, chosen, self.pending, why
                )
            else:
                selection = self._select_lucb(local_models, server_round)
                self.local_acc = {}
                for client_id, observations in self._latest_observations.items():
                    if observations:
                        correct = sum(
                            item["prediction"] == item["label"] for item in observations.values()
                        )
                        self.local_acc[client_id] = correct / len(observations)
                self.local_acc_hist.append({name(cid): acc for cid, acc in self.local_acc.items()})
                delivery = self._lucb_delivery_state(selection)
                if delivery["status"] == "eligible":
                    self.pending, explanation_evaluations = self._build_lucb_feedback(
                        local_models, selection
                    )
                    delivered_entries = sum(len(entries) for entries in self.pending.values())
                    delivery = {
                        "status": "delivered" if delivered_entries else "not_delivered",
                        "reason": None if delivered_entries else "no_eligible_selected_errors",
                        "entry_count": delivered_entries,
                        "selection_status": selection.status,
                        "selection_stop_reason": selection.stop_reason,
                    }
                else:
                    self.pending = {}
                    explanation_evaluations = 0
                self._write_lucb_audit(
                    server_round, selection, explanation_evaluations, delivery
                )
                print(f"[R{server_round}] LUCB {selection.status}: {selection.stop_reason}; "
                      f"selected {[name(client_id) for client_id in selection.selected_client_ids]}")
        else:
            rng = random.Random(server_round)
            chosen = rng.sample(sorted(local_models), min(self.n_clients, len(local_models)))
            print(f"[R{server_round}] GENERATING DATA, no feedback ({why}) -> {len(chosen)} "
                  f"random clients: {[name(c) for c in chosen]}")
            self.pending = {cid: {"__gen_budget__": {"n_classes": self.n_classes,
                                                     "n_per_class": self.n_per_class}}
                            for cid in chosen}

        produced = sum(len(v) for v in self.pending.values())
        if produced:
            self.fb_rounds.append(server_round)
        else:
            print(f"[R{server_round}] no eligible samples -> not counted as a data generation round")
            self.pending = {}

        if self.use_feedback:
            os.makedirs(SERVER_LOG_DIR, exist_ok=True)
            with open(os.path.join(SERVER_LOG_DIR,
                                   f"feedback_round_{server_round}.json"), "w") as f:
                json.dump({
                    client_id: [
                        {key: value for key, value in entry.items() if key not in {"values", "sample_id"}}
                        for entry in entries
                    ]
                    for client_id, entries in self.pending.items()
                }, f, indent=2)
        return params, metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        if not self.pending:
            return ins

        out = []
        for proxy, fit_ins in ins:
            cfg = dict(fit_ins.config)
            fbs = self.pending.get(proxy.cid)
            if fbs and not self.use_feedback:
                b = fbs["__gen_budget__"]          
                cfg.update({"gen_budget": json.dumps(b), "diff_steps": self.diff_steps})
                print(f"[R{server_round}] -> {self.names.get(proxy.cid, proxy.cid)}: generate "
                      f"{b['n_classes']} x {b['n_per_class']} images, classes chosen by the "
                      f"client (no feedback)")
            elif fbs:
                cfg.update({"feedback": json.dumps(fbs), "n_per_class": self.n_per_class,
                            "mode": self.gen_mode, "top_m": self.top_m,
                            "diff_steps": self.diff_steps,
                            "feedback_protocol": self.feedback_protocol,
                            "allow_raw_feedback": self.allow_raw_feedback})
                print(f"[R{server_round}] -> {self.names.get(proxy.cid, proxy.cid)}: {len(fbs)} feedback elements "
                      f"({len(json.dumps(fbs)) / 1024:.0f} KB)")
            out.append((proxy, FitIns(fit_ins.parameters, cfg)))

        self.pending = {}      # feedback is single-use
        return out


def weighted_average(metrics):
    total = sum(n for n, _ in metrics) or 1
    keys = {k for _, m in metrics for k, v in m.items() if isinstance(v, (int, float))}
    return {k: sum(n * m[k] for n, m in metrics if k in m) / total for k in keys}


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FLGenAI server")
    p.add_argument("--target", required=True,
                   help="subset directory used as target set D^t ")
    p.add_argument("--clients", type=int, default=9,
                   help="minimum number of clients (10 subsets, 1 as target -> 9)")
    p.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    # §2 — trigger conditions
    p.add_argument("--min-rounds", type=int, default=5)
    p.add_argument("--acc-drop-tol", type=float, default=0.0)
    p.add_argument("--plateau-patience", type=int, default=3)
    p.add_argument("--plateau-delta", type=float, default=2e-3)
    p.add_argument("--overfit-gap", type=float, default=0.15)
    # §3.4 — budget
    p.add_argument("--n-clients", type=int, default=N_CLIENTS_FB)
    p.add_argument("--n-classes", type=int, default=N_CLASSES_FB)
    p.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    p.add_argument("--diff-steps", type=int, default=500,
                   help="number of reverse diffusion steps on the client")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--top-m", type=int, default=TOP_M)
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--selection-mode", choices=["lucb", "legacy_full_scan"], default="lucb",
                   help="client-selection strategy for feedback events")
    p.add_argument("--lucb-budget-fraction", type=float, default=0.25,
                   help="fraction of a full client scan available to LUCB")
    p.add_argument("--lucb-batch-size", type=int, default=64,
                   help="number of target samples per LUCB minibatch")
    p.add_argument("--lucb-top-k", type=int, default=0,
                   help="number of weak clients to select; 0 keeps the image default")
    p.add_argument("--lucb-epsilon", type=float, default=0.02,
                   help="LUCB boundary tolerance for client error rates")
    p.add_argument("--lucb-delta", type=float, default=0.05,
                   help="family-wise confidence error budget")
    p.add_argument("--weak-class-accuracy", type=float, default=0.85,
                   help="upper Wilson accuracy threshold for a weak class")
    p.add_argument("--lucb-prioritization", choices=["strict", "decay_priority"], default="strict",
                   help="strict statistical scheduling or experimental decayed priority")
    p.add_argument("--feedback-protocol", choices=["abstract", "legacy"], default="abstract",
                   help="abstract feedback omits server-side pixels")
    p.add_argument("--allow-raw-feedback", action="store_true",
                   help="explicitly permit legacy payloads with raw target patches")
    p.add_argument("--allow-partial-feedback", action="store_true",
                   help="deliver feedback to identified weak classes even if budget was exhausted before full separation")
    p.add_argument("--force-feedback", action="store_true",
                   help="force feedback events for controlled smoke experiments")
    p.add_argument("--validate-config", action="store_true",
                   help="validate paths and protocol settings without starting Flower")
    p.add_argument("--mode", default=GEN_MODE, choices=["refine", "repaint", "paper"],
                   help="image generation mode forced on clients")
    p.add_argument("--no-feedback", action="store_true",
                   help="generate images without send feedback")
    p.add_argument("--method", default="proposed", choices=METHODS,
                   help="proposed | nofeedback | fedavg | fedcog")
    # FedCOG (only used with --method fedcog)
    p.add_argument("--fedcog-start", type=int, default=FEDCOG_START)
    p.add_argument("--fedcog-n", type=int, default=FEDCOG_N)
    p.add_argument("--fedcog-steps", type=int, default=FEDCOG_STEPS)
    p.add_argument("--fedcog-lr", type=float, default=FEDCOG_LR)
    p.add_argument("--fedcog-adv", type=float, default=FEDCOG_ADV)
    p.add_argument("--fedcog-T", type=float, default=FEDCOG_T)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-dir", default=SERVER_LOG_DIR)
    return p.parse_args(argv)


def main(argv=None):
    global DEVICE, SERVER_LOG_DIR

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = parse_args(argv)
    if args.no_feedback:
        args.method = "nofeedback"
    if args.feedback_protocol == "legacy" and not args.allow_raw_feedback:
        raise ValueError("Legacy feedback requires --allow-raw-feedback")
    if args.feedback_protocol == "legacy":
        warnings.warn(
            "Legacy raw-patch feedback is enabled. Raw target patches are transmitted at runtime, but audit logs remain sanitized.",
            RuntimeWarning,
        )
    if args.feedback_protocol == "abstract" and args.mode != "refine":
        raise ValueError("Abstract feedback requires --mode refine")
    if not 0 < args.lucb_budget_fraction <= 1:
        raise ValueError("--lucb-budget-fraction must be in (0, 1]")
    if args.lucb_batch_size <= 0:
        raise ValueError("--lucb-batch-size must be positive")
    if args.lucb_top_k < 0:
        raise ValueError("--lucb-top-k must be zero (automatic) or positive")
    if args.lucb_epsilon < 0:
        raise ValueError("--lucb-epsilon must be non-negative")
    if not 0 < args.lucb_delta < 1:
        raise ValueError("--lucb-delta must be in (0, 1)")
    if not 0 < args.weak_class_accuracy < 1:
        raise ValueError("--weak-class-accuracy must be in (0, 1)")
    if args.validate_config:
        if not os.path.isdir(args.target):
            raise FileNotFoundError(f"Target subset does not exist: {args.target}")
        print("[server] configuration validation passed")
        return
    set_seed(args.seed)
    DEVICE = get_device(args.device)
    SERVER_LOG_DIR = args.log_dir
    os.makedirs(SERVER_LOG_DIR, exist_ok=True)
    print(f"[server] device={DEVICE} rounds={args.rounds} clients={args.clients} "
          f"method={args.method}")

    def fit_config(server_round: int):
        """Sent to every client each round; tells it which method to run."""
        cfg = {"method": args.method, "server_round": server_round}
        if args.method == "fedcog":
            cfg.update({"fedcog_start": args.fedcog_start, "fedcog_n": args.fedcog_n,
                        "fedcog_steps": args.fedcog_steps, "fedcog_lr": args.fedcog_lr,
                        "fedcog_adv": args.fedcog_adv, "fedcog_T": args.fedcog_T})
        return cfg

    target_set = load_subset(args.target)

    init = resnet18(NUM_CLASSES)
    initial_parameters = ndarrays_to_parameters(
        [v.cpu().numpy() for v in init.state_dict().values()])

    strategy = FeedbackFedAvg(
        test_set=target_set,
        target_layer_of=lambda m: m.layer3,   # 8x8 feature map -> 8x8 CAM grid
        initial_parameters=initial_parameters,
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=args.clients, min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        min_rounds=args.min_rounds, acc_drop_tol=args.acc_drop_tol,
        plateau_patience=args.plateau_patience, plateau_delta=args.plateau_delta,
        overfit_gap=args.overfit_gap,
        n_clients=args.n_clients, n_classes=args.n_classes,
        n_per_class=args.n_per_class, diff_steps=args.diff_steps,
        top_k=args.top_k, top_m=args.top_m,
        grid=args.grid, gen_mode=args.mode,
        use_feedback=args.method != "nofeedback", method=args.method,
        selection_mode=args.selection_mode,
        lucb_budget_fraction=args.lucb_budget_fraction,
        lucb_batch_size=args.lucb_batch_size,
        lucb_top_k=None if args.lucb_top_k == 0 else args.lucb_top_k,
        lucb_epsilon=args.lucb_epsilon,
        lucb_delta=args.lucb_delta,
        weak_class_accuracy=args.weak_class_accuracy,
        lucb_prioritization=args.lucb_prioritization,
        seed=args.seed,
        feedback_protocol=args.feedback_protocol,
        allow_raw_feedback=args.allow_raw_feedback,
        force_feedback=args.force_feedback,
        allow_partial_feedback=args.allow_partial_feedback,
        on_fit_config_fn=fit_config,
        device=DEVICE,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"[server] listening on {SERVER_ADDRESS} ...")
    hist = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print(f"\n[server] data generation rounds: {strategy.fb_rounds}")
    print(f"[server] ({len(strategy.fb_rounds)}/{args.rounds} rounds — paper: 16-18/50)")
    with open(os.path.join(SERVER_LOG_DIR, "summary.json"), "w") as f:
        json.dump({"method": args.method,
                   "selection_mode": args.selection_mode,
                   "feedback_protocol": args.feedback_protocol,
                   "lucb": {
                       "budget_fraction": args.lucb_budget_fraction,
                       "batch_size": args.lucb_batch_size,
                       "top_k": strategy.lucb_config.resolved_top_k(args.clients),
                       "epsilon": args.lucb_epsilon,
                       "delta": args.lucb_delta,
                       "weak_class_accuracy": args.weak_class_accuracy,
                       "prioritization": args.lucb_prioritization,
                   },
                   "acc_hist": strategy.acc_hist,                 # global model on D^t
                   "local_acc_hist": strategy.local_acc_hist,     # each local model on D^t
                   "fb_rounds": strategy.fb_rounds,
                   "client_val_loss": hist.losses_distributed,    # weighted mean of client val
                   "client_val_acc": hist.metrics_distributed.get("accuracy", []),
                   "client_train": hist.metrics_distributed_fit}, f, indent=2)


if __name__ == "__main__":
    main()
