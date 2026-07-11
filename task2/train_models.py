"""
train_models.py
===============
Training loops for ANN and SNN (TET), with representation drift tracking.
Both use cross-entropy loss + cosine LR schedule + Adam optimizer.
SNN uses TET loss (uniform CE across timesteps).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TET loss (SNN)
# ---------------------------------------------------------------------------
def tet_loss(outputs, labels, criterion, means=1.0, lamb=1e-3):
    """outputs: [B, T, C]"""
    T = outputs.shape[1]
    loss_ce = sum(criterion(outputs[:, t], labels) for t in range(T)) / T
    y = torch.zeros_like(outputs).fill_(means)
    loss_mse = nn.functional.mse_loss(outputs, y)
    return (1 - lamb) * loss_ce + lamb * loss_mse


# ---------------------------------------------------------------------------
# ANN training
# ---------------------------------------------------------------------------
def train_ann_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def eval_ann(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += model(images).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def run_ann(model, train_loader, val_loader, probe_images, device,
            epochs=30, lr=1e-3):
    from representation_drift_tracker import RepresentationDriftTracker, build_probe_set
    tracker = RepresentationDriftTracker(model, "ANN", probe_images, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_ann_epoch(model, train_loader, optimizer, device)
        test_acc = eval_ann(model, val_loader, device)
        scheduler.step()
        tracker.step(epoch, train_acc, test_acc, loss)

    return tracker


# ---------------------------------------------------------------------------
# SNN (TET) training
# ---------------------------------------------------------------------------
def train_snn_epoch(model, loader, optimizer, device, means=1.0, lamb=1e-3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)         # [B, T, C]
        loss = tet_loss(outputs, labels, criterion, means, lamb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += outputs.mean(1).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def eval_snn(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += model(images).mean(1).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def run_snn(model, train_loader, val_loader, probe_images, device,
            epochs=30, lr=1e-3, means=1.0, lamb=1e-3):
    from representation_drift_tracker import RepresentationDriftTracker
    tracker = RepresentationDriftTracker(model, "SNN_TET", probe_images, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss, train_acc = train_snn_epoch(model, train_loader, optimizer, device, means, lamb)
        test_acc = eval_snn(model, val_loader, device)
        scheduler.step()
        tracker.step(epoch, train_acc, test_acc, loss)

    return tracker
