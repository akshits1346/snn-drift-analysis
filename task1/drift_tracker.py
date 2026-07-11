"""
DriftTracker
============
Tracks two drift signals per epoch:

  1. Weight drift (per conv layer):
       || W_t - W_{t-1} ||_2  (L2 norm of weight delta)

  2. Firing rate drift (per LIF layer):
       | avg_firing_rate_t - avg_firing_rate_{t-1} |
       where avg_firing_rate is the mean spike probability across the
       validation set at that epoch.

Both signals measure how much the model *changes* epoch-by-epoch,
giving a clear picture of training stability vs. convergence speed
for TET vs. event-based backprop.
"""

import copy
import json
from collections import defaultdict

import torch
import numpy as np


class DriftTracker:
    def __init__(self, model, algo_name: str):
        self.algo = algo_name
        self.model = model

        # snapshots: layer_name → weight tensor (CPU, detached)
        self._prev_weights: dict[str, torch.Tensor] = {}
        self._prev_fire_rates: dict[str, float] = {}

        # logs
        self.weight_drift: dict[str, list[float]] = defaultdict(list)
        self.fire_rate_drift: dict[str, list[float]] = defaultdict(list)
        self.fire_rates: dict[str, list[float]] = defaultdict(list)   # absolute rates too
        self.epoch_log: list[dict] = []   # per-epoch summary

        # take initial snapshot before epoch 1
        self._snapshot_weights()

    # ------------------------------------------------------------------
    # Weight snapshot helpers
    # ------------------------------------------------------------------
    def _snapshot_weights(self):
        for name, module in self.model.named_conv_layers():
            self._prev_weights[name] = module.weight.detach().cpu().clone()

    def _compute_weight_drift(self) -> dict[str, float]:
        drifts = {}
        for name, module in self.model.named_conv_layers():
            curr = module.weight.detach().cpu()
            prev = self._prev_weights.get(name)
            if prev is not None:
                drifts[name] = (curr - prev).norm(p=2).item()
            else:
                drifts[name] = 0.0
        return drifts

    # ------------------------------------------------------------------
    # Firing-rate measurement: run the model on val_loader, collect
    # average spike counts per LIF layer via forward hooks.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def measure_firing_rates(self, val_loader, device, max_batches: int = 10) -> dict[str, float]:
        self.model.eval()
        accum: dict[str, list] = defaultdict(list)
        handles = []

        for name, lif in self.model.named_lif_layers():
            def make_hook(n):
                def hook(module, inp, out):
                    # out: [B, T, ...] — mean spike rate averaged over B,T,spatial
                    rate = out.float().mean().item()
                    accum[n].append(rate)
                return hook
            handles.append(lif.register_forward_hook(make_hook(name)))

        for i, (images, _) in enumerate(val_loader):
            if i >= max_batches:
                break
            images = images.to(device)
            _ = self.model(images)

        for h in handles:
            h.remove()

        self.model.train()
        return {name: float(np.mean(vals)) for name, vals in accum.items()}

    # ------------------------------------------------------------------
    # Call once per epoch AFTER the training step
    # ------------------------------------------------------------------
    def step(self, epoch: int, val_loader, device, train_acc: float, test_acc: float, loss: float):
        # --- weight drift ---
        w_drifts = self._compute_weight_drift()
        for name, d in w_drifts.items():
            self.weight_drift[name].append(d)
        self._snapshot_weights()

        # --- firing rate drift ---
        curr_rates = self.measure_firing_rates(val_loader, device)
        fr_drifts = {}
        for name, rate in curr_rates.items():
            self.fire_rates[name].append(rate)
            prev = self._prev_fire_rates.get(name)
            drift = abs(rate - prev) if prev is not None else 0.0
            fr_drifts[name] = drift
            self.fire_rate_drift[name].append(drift)
        self._prev_fire_rates = dict(curr_rates)

        # --- per-epoch summary ---
        summary = {
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "loss": loss,
            "weight_drift": dict(w_drifts),
            "fire_rate_drift": dict(fr_drifts),
            "fire_rates": dict(curr_rates),
            # aggregate across layers
            "mean_weight_drift": float(np.mean(list(w_drifts.values()))),
            "mean_fire_rate_drift": float(np.mean(list(fr_drifts.values()))),
        }
        self.epoch_log.append(summary)
        return summary

    # ------------------------------------------------------------------
    # Serialise full log to JSON
    # ------------------------------------------------------------------
    def save(self, path: str):
        data = {
            "algo": self.algo,
            "epochs": self.epoch_log,
            "weight_drift_by_layer": dict(self.weight_drift),
            "fire_rate_drift_by_layer": dict(self.fire_rate_drift),
            "fire_rates_by_layer": dict(self.fire_rates),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DriftTracker] Saved log to {path}")
