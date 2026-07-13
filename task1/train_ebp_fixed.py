"""
train_ebp.py  (FIXED VERSION)
==============================
Event-Based Backprop training loop — fixed to address the silent neuron
collapse observed in the original run (EBP stuck at ~8% / random guessing).

DIAGNOSIS OF ORIGINAL FAILURE:
  The EBP count+ loss only sends gradient to neurons that already spiked.
  If neurons go silent early (which happens easily at random init with a
  high threshold), they receive zero gradient forever and never recover.
  This is the classic "dead neuron" problem, amplified by EBP's design.

THREE FIXES APPLIED:
  1. Lower LIF threshold (1.0 → 0.5)
     Neurons fire more easily at random init, preventing cold-start silence.

  2. TET warmup for first 5 epochs
     Train with TET (uniform gradient across all timesteps) first to get
     neurons firing reliably, THEN switch to EBP. This avoids the dead
     neuron problem entirely at the start of training.

  3. Tighter gradient clipping (1000 → 1.0)
     The original clip of 1000 is effectively no clipping. Exploding
     gradients in early epochs were killing neurons. Clip at 1.0 stabilises
     early training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from drift_tracker import DriftTracker


# ---------------------------------------------------------------------------
# TET loss (used for warmup phase)
# ---------------------------------------------------------------------------
def tet_loss(outputs, labels, criterion, means=1.0, lamb=1e-3):
    T = outputs.shape[1]
    loss_ce = sum(criterion(outputs[:, t], labels) for t in range(T)) / T
    y = torch.zeros_like(outputs).fill_(means)
    loss_mse = nn.functional.mse_loss(outputs, y)
    return (1 - lamb) * loss_ce + lamb * loss_mse


# ---------------------------------------------------------------------------
# EBP count+ loss (fixed)
# ---------------------------------------------------------------------------
class CountPlusLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, labels, desired_count=4, undesired_count=1):
        T, B, C = outputs.shape[1], outputs.shape[0], outputs.shape[2]
        out_count = outputs.sum(dim=1)   # [B, C]

        target = torch.ones_like(out_count) * undesired_count
        for i in range(B):
            target[i, labels[i]] = desired_count

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
        B, C = out_count.shape

        sm = F.softmax(out_count, dim=1)
        correct_idx = target.argmax(dim=1)
        sm[torch.arange(B, device=outputs.device), correct_idx] -= 1
        grad_count = grad_output * sm / B

        # Event conditioning: zero grad for silent neurons
        has_spiked = (out_count > 0).float()
        grad_count = grad_count * has_spiked

        spike_norm = out_count.clamp(min=1)
        grad_per_t = (grad_count / spike_norm).unsqueeze(1)
        grad_output_full = grad_per_t.expand_as(outputs) * outputs

        return grad_output_full, None, None, None


def ebp_loss(outputs, labels, desired_count=4, undesired_count=1):
    return CountPlusLoss.apply(outputs, labels, desired_count, undesired_count)


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------
def train_one_epoch_tet(model, loader, optimizer, device):
    """TET step used during warmup."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = tet_loss(outputs, labels, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # FIX 3
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += outputs.mean(1).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def train_one_epoch_ebp(model, loader, optimizer, device,
                        desired_count=4, undesired_count=1):
    """EBP step used after warmup."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = ebp_loss(outputs, labels, desired_count, undesired_count)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # FIX 3
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += outputs.sum(1).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += model(images).sum(1).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Full EBP training run with fixes
# ---------------------------------------------------------------------------
def run_ebp(model, train_loader, val_loader, device, epochs=30,
            lr=1e-3, desired_count=4, undesired_count=1, warmup_epochs=5):
    """
    Trains model with fixed EBP:
      - Epochs 1..warmup_epochs: TET warmup to get neurons firing
      - Epochs warmup_epochs+1..epochs: EBP count+ loss
    """
    # FIX 1: lower threshold on all LIF neurons
    from snn_model import LIFSpike
    for m in model.modules():
        if isinstance(m, LIFSpike):
            m.thresh = 0.5

    tracker = DriftTracker(model, algo_name="EBP_fixed")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        # FIX 2: TET warmup for first N epochs
        if epoch <= warmup_epochs:
            loss, train_acc = train_one_epoch_tet(model, train_loader, optimizer, device)
            phase = "warmup"
        else:
            loss, train_acc = train_one_epoch_ebp(model, train_loader, optimizer, device,
                                                   desired_count, undesired_count)
            phase = "EBP"

        test_acc = evaluate(model, val_loader, device)
        scheduler.step()

        summary = tracker.step(epoch, val_loader, device, train_acc, test_acc, loss)

        print(
            f"[EBP_fixed/{phase}] Epoch {epoch:3d}/{epochs} | "
            f"loss={loss:.4f} | train={train_acc:.1f}% | test={test_acc:.1f}% | "
            f"Δw={summary['mean_weight_drift']:.5f} | Δfr={summary['mean_fire_rate_drift']:.5f}"
        )

    return tracker
