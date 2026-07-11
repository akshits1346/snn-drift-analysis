"""
main_experiment.py
==================
Trains the same SmallSNN architecture on CIFAR-10 with both:
  - TET  (Temporal Efficient Training)
  - EBP  (Event-Based Backprop / count+ loss)

Records weight drift and firing-rate drift at every epoch.
Saves:
  - results/tet_drift.json
  - results/ebp_drift.json
  - results/drift_summary.json   (combined, for easy comparison)

Usage:
    python main_experiment.py [--epochs N] [--T T] [--batch_size B] [--seed S]

Recommended quick run: --epochs 20 --T 4
Full run (matches paper scale): --epochs 50 --T 4
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from snn_model import SmallSNN
from train_tet import run_tet
from train_ebp import run_ebp


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CIFAR-10 data loaders
# ---------------------------------------------------------------------------
def get_cifar10(batch_size: int, data_root: str = "./data"):
    """Standard CIFAR-10 with augmentation for train, plain for val."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tf)
    val_set   = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=val_tf)

    num_workers = 2
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Merge + summarise both trackers
# ---------------------------------------------------------------------------
def build_summary(tet_tracker, ebp_tracker) -> dict:
    """
    Build a side-by-side comparison dict that's easy to read
    and easy to plot with plot_drift.py.
    """
    def extract(tracker):
        epochs = [e["epoch"]              for e in tracker.epoch_log]
        return {
            "epochs":              epochs,
            "train_acc":           [e["train_acc"]            for e in tracker.epoch_log],
            "test_acc":            [e["test_acc"]             for e in tracker.epoch_log],
            "loss":                [e["loss"]                 for e in tracker.epoch_log],
            "mean_weight_drift":   [e["mean_weight_drift"]    for e in tracker.epoch_log],
            "mean_fire_rate_drift":[e["mean_fire_rate_drift"] for e in tracker.epoch_log],
            "weight_drift_layers": dict(tracker.weight_drift),
            "fire_rate_drift_layers": dict(tracker.fire_rate_drift),
            "fire_rates_layers":   dict(tracker.fire_rates),
        }

    return {"TET": extract(tet_tracker), "EBP": extract(ebp_tracker)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SNN drift experiment: TET vs EBP")
    parser.add_argument("--epochs",     type=int, default=30,  help="Training epochs per algorithm")
    parser.add_argument("--T",          type=int, default=4,   help="SNN simulation timesteps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed",       type=int, default=42,  help="Random seed")
    parser.add_argument("--data_root",  type=str, default="./data", help="CIFAR-10 download path")
    parser.add_argument("--out_dir",    type=str, default="./results", help="Output directory")
    # TET-specific
    parser.add_argument("--tet_means",  type=float, default=1.0,  help="TET target mean")
    parser.add_argument("--tet_lamb",   type=float, default=1e-3, help="TET lambda (MSE weight)")
    # EBP-specific
    parser.add_argument("--ebp_desired",   type=int, default=4, help="EBP desired spike count")
    parser.add_argument("--ebp_undesired", type=int, default=1, help="EBP undesired spike count")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  SNN DRIFT EXPERIMENT: TET vs Event-Based Backprop")
    print(f"  epochs={args.epochs}, T={args.T}, batch={args.batch_size}, seed={args.seed}")
    print("=" * 65)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_loader, val_loader = get_cifar10(args.batch_size, args.data_root)

    # -------------------------------------------------------------------
    # Run 1: TET
    # -------------------------------------------------------------------
    print("\n" + "─" * 50)
    print("  PHASE 1 — TET (Temporal Efficient Training)")
    print("─" * 50)
    seed_all(args.seed)                             # identical init for fair comparison
    model_tet = SmallSNN(in_channels=3, n_classes=10, T=args.T).to(device)

    tet_tracker = run_tet(
        model_tet, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        means=args.tet_means,
        lamb=args.tet_lamb,
    )
    tet_path = os.path.join(args.out_dir, "tet_drift.json")
    tet_tracker.save(tet_path)

    # -------------------------------------------------------------------
    # Run 2: EBP
    # -------------------------------------------------------------------
    print("\n" + "─" * 50)
    print("  PHASE 2 — EBP (Event-Based Backprop, count+ loss)")
    print("─" * 50)
    seed_all(args.seed)                             # identical init
    model_ebp = SmallSNN(in_channels=3, n_classes=10, T=args.T).to(device)

    ebp_tracker = run_ebp(
        model_ebp, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        desired_count=args.ebp_desired,
        undesired_count=args.ebp_undesired,
    )
    ebp_path = os.path.join(args.out_dir, "ebp_drift.json")
    ebp_tracker.save(ebp_path)

    # -------------------------------------------------------------------
    # Combined summary
    # -------------------------------------------------------------------
    summary = build_summary(tet_tracker, ebp_tracker)
    summary_path = os.path.join(args.out_dir, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Combined summary saved to {summary_path}")

    # -------------------------------------------------------------------
    # Quick text comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  FINAL COMPARISON")
    print("=" * 65)
    for algo, data in summary.items():
        last = -1
        print(
            f"  {algo:4s}  |  test_acc={data['test_acc'][last]:.1f}%  |"
            f"  mean_Δw={data['mean_weight_drift'][last]:.5f}  |"
            f"  mean_Δfr={data['mean_fire_rate_drift'][last]:.5f}"
        )
    print()
    print("  Run `python plot_drift.py` to generate figures.")


if __name__ == "__main__":
    main()
