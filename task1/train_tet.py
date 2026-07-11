"""
TET training loop
=================
Temporal Efficient Training (Deng et al., ICLR 2022)

Loss: L_TET = (1/T) * sum_t CE(output_t, y)  +  lambda * MSE(output, means)

The key idea is that BPTT gradients are re-weighted uniformly across all
timesteps (every timestep contributes to the loss), which avoids
vanishing temporal gradients by not just using the final timestep.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from drift_tracker import DriftTracker


# ---------------------------------------------------------------------------
# TET loss (from functions.py in the original repo)
# ---------------------------------------------------------------------------
def tet_loss(outputs, labels, criterion, means: float = 1.0, lamb: float = 1e-3):
    """
    outputs: [B, T, n_classes]
    labels:  [B]
    Returns scalar loss.
    """
    T = outputs.shape[1]
    loss_ce = sum(criterion(outputs[:, t], labels) for t in range(T)) / T
    if lamb != 0:
        y = torch.zeros_like(outputs).fill_(means)
        loss_mse = nn.functional.mse_loss(outputs, y)
    else:
        loss_mse = 0.0
    return (1 - lamb) * loss_ce + lamb * loss_mse


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, means=1.0, lamb=1e-3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)          # [B, T, C]
        loss = tet_loss(outputs, labels, criterion, means, lamb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.mean(1).argmax(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).mean(1)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Full TET training run with drift tracking
# ---------------------------------------------------------------------------
def run_tet(model, train_loader, val_loader, device, epochs=30,
            lr=1e-3, means=1.0, lamb=1e-3):
    """
    Trains model with TET loss and records drift at every epoch.
    Returns the populated DriftTracker.
    """
    tracker = DriftTracker(model, algo_name="TET")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, means, lamb)
        test_acc = evaluate(model, val_loader, device)
        scheduler.step()

        summary = tracker.step(epoch, val_loader, device, train_acc, test_acc, loss)

        print(
            f"[TET] Epoch {epoch:3d}/{epochs} | "
            f"loss={loss:.4f} | train={train_acc:.1f}% | test={test_acc:.1f}% | "
            f"Δw={summary['mean_weight_drift']:.5f} | Δfr={summary['mean_fire_rate_drift']:.5f}"
        )

    return tracker
