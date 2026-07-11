"""
Event-Based Backprop (EBP) training loop
=========================================
Zhu et al., NeurIPS 2023 — "Exploring Loss Functions for Time-based
Training Strategy in Spiking Neural Networks"

Key idea: rather than propagating gradients uniformly across timesteps
(as TET does), EBP conditions the gradient on *when* spikes actually
occur. The gradient for each neuron is modulated by whether/when an
event (spike) happened, making the learning signal inherently temporal
and event-driven.

We implement the "count+" loss here — spike count cross-entropy with a
target count surplus for the correct class — which is the primary
variant studied in the paper and is a clean drop-in loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from drift_tracker import DriftTracker


# ---------------------------------------------------------------------------
# EBP loss: count+ (spike-count cross-entropy with margin)
# ---------------------------------------------------------------------------
class CountPlusLoss(torch.autograd.Function):
    """
    'count+' loss from the NeurIPS 2023 paper.

    outputs: [B, T, C]  — spike tensor
    labels:  [B]

    Gradient is non-zero only for neurons that have already spiked,
    making it an event-conditioned update — the hallmark of event-based
    learning.
    """
    @staticmethod
    def forward(ctx, outputs, labels, desired_count=4, undesired_count=1):
        # outputs: [B, T, C]
        T, B, C = outputs.shape[1], outputs.shape[0], outputs.shape[2]
        out_count = outputs.sum(dim=1)       # [B, C]  total spikes

        target = torch.ones_like(out_count) * undesired_count
        for i in range(B):
            target[i, labels[i]] = desired_count

        # saturate: correct class capped at desired, others floored at undesired
        out_count_sat = out_count.clone()
        mask_des = (target == desired_count)
        mask_und = (target == undesired_count)
        out_count_sat[mask_des & (out_count > desired_count)] = desired_count
        out_count_sat[mask_und & (out_count < undesired_count)] = undesired_count

        ctx.save_for_backward(outputs, out_count, target,
                              torch.tensor([T, desired_count, undesired_count]))
        return F.cross_entropy(out_count_sat, labels)

    @staticmethod
    def backward(ctx, grad_output):
        outputs, out_count, target, params = ctx.saved_tensors
        T = params[0].item()
        desired_count = params[1].item()

        # CE gradient w.r.t. out_count_sat, then chain through saturation
        B, C = out_count.shape
        sm = F.softmax(out_count, dim=1)
        idx = torch.arange(B, device=outputs.device)
        sm[idx, target[:, 0].long() if target.dim() > 1 else target.argmax(1)] -= 1
        grad_count = grad_output * sm / B             # [B, C]

        # --- event conditioning: zero out grad for neurons that haven't spiked ---
        has_spiked = (out_count > 0).float()          # [B, C]
        grad_count = grad_count * has_spiked

        # distribute evenly across timesteps where the neuron spiked
        spike_norm = out_count.clamp(min=1)           # avoid div-by-zero
        grad_per_t = (grad_count / spike_norm).unsqueeze(1)   # [B, 1, C]
        grad_output_full = grad_per_t.expand_as(outputs) * outputs  # [B, T, C]

        return grad_output_full, None, None, None


def ebp_loss(outputs, labels, desired_count=4, undesired_count=1):
    """outputs: [B, T, C],  labels: [B]"""
    return CountPlusLoss.apply(outputs, labels, desired_count, undesired_count)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device,
                    desired_count=4, undesired_count=1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)     # [B, T, C]
        loss = ebp_loss(outputs, labels, desired_count, undesired_count)
        loss.backward()

        # gradient clipping (mirroring the original repo's clip_grad_norm_)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.sum(1).argmax(1)     # classify by total spike count
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).sum(1)   # total spike count readout
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Full EBP training run with drift tracking
# ---------------------------------------------------------------------------
def run_ebp(model, train_loader, val_loader, device, epochs=30,
            lr=1e-3, desired_count=4, undesired_count=1):
    """
    Trains model with EBP (count+) loss and records drift at every epoch.
    Returns the populated DriftTracker.
    """
    tracker = DriftTracker(model, algo_name="EBP")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device,
                                          desired_count, undesired_count)
        test_acc = evaluate(model, val_loader, device)
        scheduler.step()

        summary = tracker.step(epoch, val_loader, device, train_acc, test_acc, loss)

        print(
            f"[EBP] Epoch {epoch:3d}/{epochs} | "
            f"loss={loss:.4f} | train={train_acc:.1f}% | test={test_acc:.1f}% | "
            f"Δw={summary['mean_weight_drift']:.5f} | Δfr={summary['mean_fire_rate_drift']:.5f}"
        )

    return tracker
