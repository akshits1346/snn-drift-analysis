"""
representation_drift_tracker.py
================================
Measures REPRESENTATION DRIFT — how much the internal feature vectors
that a network produces for the same images change between epochs.

This is fundamentally different from weight drift (Task 1):
  - Weight drift: how much do the parameters move?
  - Representation drift: how much do the *outputs of each layer* move
    for a fixed set of input images?

Why this matters for the research question:
  Surrogate gradients are approximations. Even if weights move similarly,
  the approximation error could cause representations to be less stable
  (more drift) in SNNs vs ANNs. Separately, SNN sparsity might
  *counteract* this by dampening the propagation of gradient errors.

Metrics tracked per epoch per layer:
  1. CKA drift        — 1 - CKA(H_t, H_{t-1})
                        CKA (Centered Kernel Alignment) measures how
                        similar two sets of representations are, invariant
                        to rotation and scaling. Standard in deep learning
                        representation analysis.
  2. L2 repr drift    — mean ||h_t(x) - h_{t-1}(x)||_2 across the probe set
                        Raw magnitude of representation change.
  3. Sparsity         — fraction of zero/silent activations per layer
  4. Sparsity-drift correlation — computed at end of run
"""

import json
from collections import defaultdict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------
def center(K):
    """Double-center a kernel matrix."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H


def linear_CKA(X, Y):
    """
    Linear CKA between two representation matrices X, Y: [n_samples, d].
    Returns scalar in [0, 1]. 1 = identical representations (up to rotation).
    """
    X = X.float()
    Y = Y.float()
    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T
    Kc = center(K)
    Lc = center(L)
    hsic_xy = (Kc * Lc).sum()
    hsic_xx = (Kc * Kc).sum().sqrt()
    hsic_yy = (Lc * Lc).sum().sqrt()
    denom = hsic_xx * hsic_yy
    if denom < 1e-10:
        return torch.tensor(1.0)
    return hsic_xy / denom


# ---------------------------------------------------------------------------
# Probe set: a fixed subset of validation images used every epoch
# ---------------------------------------------------------------------------
def build_probe_set(val_loader, n_samples=512, device='cpu'):
    """
    Extract a fixed set of images from the val loader.
    These same images are used every epoch to measure representation drift.
    Using a fixed set is critical — drift must be measured on the same inputs.
    """
    images_list, labels_list = [], []
    total = 0
    for imgs, labels in val_loader:
        images_list.append(imgs)
        labels_list.append(labels)
        total += imgs.shape[0]
        if total >= n_samples:
            break
    images = torch.cat(images_list, dim=0)[:n_samples]
    labels = torch.cat(labels_list, dim=0)[:n_samples]
    return images.to(device), labels.to(device)


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------
class RepresentationDriftTracker:
    def __init__(self, model, algo_name: str, probe_images: torch.Tensor, device):
        self.model = model
        self.algo = algo_name
        self.probe_images = probe_images
        self.device = device

        # Previous epoch representations: layer -> tensor [n_probe, D]
        self._prev_reps: dict[str, torch.Tensor] = {}

        # Logs
        self.cka_drift: dict[str, list[float]] = defaultdict(list)      # 1 - CKA
        self.l2_drift:  dict[str, list[float]] = defaultdict(list)      # mean L2
        self.sparsity:  dict[str, list[float]] = defaultdict(list)      # fraction silent
        self.cka_abs:   dict[str, list[float]] = defaultdict(list)      # CKA itself
        self.epoch_log: list[dict] = []

        # Take initial snapshot (epoch 0, before any training)
        self._snapshot()

    @torch.no_grad()
    def _snapshot(self):
        """Extract representations for probe set from current model state."""
        self.model.eval()
        reps = self.model.get_representations(self.probe_images)
        sparsity = self.model.get_sparsity(self.probe_images)
        self.model.train()
        self._curr_reps = {k: v.cpu() for k, v in reps.items()}
        self._curr_sparsity = sparsity
        return reps, sparsity

    def step(self, epoch: int, train_acc: float, test_acc: float, loss: float):
        """Call after each training epoch."""
        curr_reps, curr_sparsity = self._snapshot()

        cka_drifts, l2_drifts, ckas = {}, {}, {}

        for layer, curr in self._curr_reps.items():
            prev = self._prev_reps.get(layer)
            if prev is not None:
                # CKA drift
                cka_val = linear_CKA(curr, prev).item()
                cka_val = float(np.clip(cka_val, 0.0, 1.0))
                cka_drift = 1.0 - cka_val
                ckas[layer] = cka_val
                cka_drifts[layer] = cka_drift

                # L2 drift: mean over probe samples
                l2 = (curr - prev).norm(dim=1).mean().item()
                l2_drifts[layer] = l2
            else:
                ckas[layer] = 1.0
                cka_drifts[layer] = 0.0
                l2_drifts[layer] = 0.0

            self.cka_drift[layer].append(cka_drifts[layer])
            self.l2_drift[layer].append(l2_drifts.get(layer, 0.0))
            self.cka_abs[layer].append(ckas.get(layer, 1.0))
            self.sparsity[layer].append(curr_sparsity[layer])

        # Update prev
        self._prev_reps = dict(self._curr_reps)

        summary = {
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "loss": loss,
            "cka_drift": dict(cka_drifts),
            "l2_drift": dict(l2_drifts),
            "cka": dict(ckas),
            "sparsity": dict(curr_sparsity),
            "mean_cka_drift": float(np.mean(list(cka_drifts.values()))),
            "mean_l2_drift":  float(np.mean(list(l2_drifts.values()))),
            "mean_sparsity":  float(np.mean(list(curr_sparsity.values()))),
        }
        self.epoch_log.append(summary)

        print(
            f"[{self.algo}] Epoch {epoch:3d} | "
            f"acc={test_acc:.1f}% | "
            f"CKA_drift={summary['mean_cka_drift']:.5f} | "
            f"L2_drift={summary['mean_l2_drift']:.3f} | "
            f"sparsity={summary['mean_sparsity']:.3f}"
        )
        return summary

    def compute_sparsity_drift_correlation(self):
        """
        Correlate sparsity with CKA drift across epochs and layers.
        Key analysis: does higher sparsity → lower representation drift?
        """
        all_sparsity, all_cka_drift = [], []
        for layer in self.cka_drift:
            sp = self.sparsity[layer]
            cd = self.cka_drift[layer]
            n = min(len(sp), len(cd))
            all_sparsity.extend(sp[:n])
            all_cka_drift.extend(cd[:n])

        if len(all_sparsity) < 3:
            return {"pearson_r": None, "interpretation": "not enough data"}

        r = float(np.corrcoef(all_sparsity, all_cka_drift)[0, 1])
        if r < -0.3:
            interp = "Higher sparsity → LESS drift (sparsity stabilises representations)"
        elif r > 0.3:
            interp = "Higher sparsity → MORE drift (sparsity destabilises representations)"
        else:
            interp = "Sparsity and drift are weakly correlated"

        return {"pearson_r": r, "interpretation": interp}

    def save(self, path: str):
        data = {
            "algo": self.algo,
            "epochs": self.epoch_log,
            "cka_drift_by_layer": dict(self.cka_drift),
            "l2_drift_by_layer":  dict(self.l2_drift),
            "cka_by_layer":       dict(self.cka_abs),
            "sparsity_by_layer":  dict(self.sparsity),
            "sparsity_drift_correlation": self.compute_sparsity_drift_correlation(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[RepDriftTracker] Saved → {path}")
