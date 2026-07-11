"""
main_task2.py
=============
Task 2: Representation Drift in SNNs under Surrogate Gradient Training

Research questions:
  1. Do surrogate gradients cause SNN representations to drift MORE
     than equivalent ANN representations?
  2. Does the sparsity of spiking activity help COUNTERACT that drift?

Experiment:
  - Train SmallANN  (ReLU) on CIFAR-10, track representation drift
  - Train SmallSNN  (LIF + TET surrogate gradient) on CIFAR-10, track drift
  - Compare CKA drift and L2 drift per layer per epoch
  - Correlate sparsity with drift to answer question 2

Usage:
    python main_task2.py [--epochs N] [--T T] [--batch_size B]

Quick test:  python main_task2.py --epochs 5  --batch_size 64
Full run:    python main_task2.py --epochs 30 --batch_size 128
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from ann_model import SmallANN
from snn_model import SmallSNN
from train_models import run_ann, run_snn
from representation_drift_tracker import build_probe_set


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10(batch_size, data_root="./data"):
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def build_summary(ann_tracker, snn_tracker):
    def extract(t):
        return {
            "algo": t.algo,
            "epochs":           [e["epoch"]           for e in t.epoch_log],
            "test_acc":         [e["test_acc"]         for e in t.epoch_log],
            "train_acc":        [e["train_acc"]        for e in t.epoch_log],
            "loss":             [e["loss"]             for e in t.epoch_log],
            "mean_cka_drift":   [e["mean_cka_drift"]   for e in t.epoch_log],
            "mean_l2_drift":    [e["mean_l2_drift"]    for e in t.epoch_log],
            "mean_sparsity":    [e["mean_sparsity"]    for e in t.epoch_log],
            "cka_drift_layers": dict(t.cka_drift),
            "l2_drift_layers":  dict(t.l2_drift),
            "sparsity_layers":  dict(t.sparsity),
            "sparsity_drift_correlation": t.compute_sparsity_drift_correlation(),
        }
    return {"ANN": extract(ann_tracker), "SNN_TET": extract(snn_tracker)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--T",          type=int,   default=4)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--probe_size", type=int,   default=512,
                        help="Number of fixed val images used to measure drift")
    parser.add_argument("--data_root",  type=str,   default="./data")
    parser.add_argument("--out_dir",    type=str,   default="./results_task2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  TASK 2: Representation Drift — ANN vs SNN (TET)")
    print(f"  epochs={args.epochs}, T={args.T}, batch={args.batch_size}")
    print("=" * 65)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_loader, val_loader = get_cifar10(args.batch_size, args.data_root)

    # Fixed probe set — same images used every epoch for drift measurement
    seed_all(args.seed)
    probe_images, _ = build_probe_set(val_loader, n_samples=args.probe_size, device=device)
    print(f"Probe set: {probe_images.shape[0]} fixed validation images\n")

    # -----------------------------------------------------------------------
    # Run 1: ANN
    # -----------------------------------------------------------------------
    print("─" * 50)
    print("  PHASE 1 — ANN (ReLU baseline)")
    print("─" * 50)
    seed_all(args.seed)
    ann_model = SmallANN(in_channels=3, n_classes=10).to(device)
    ann_tracker = run_ann(ann_model, train_loader, val_loader, probe_images,
                          device, epochs=args.epochs, lr=args.lr)
    ann_tracker.save(os.path.join(args.out_dir, "ann_repr_drift.json"))

    # -----------------------------------------------------------------------
    # Run 2: SNN (TET)
    # -----------------------------------------------------------------------
    print("\n" + "─" * 50)
    print("  PHASE 2 — SNN with TET surrogate gradient")
    print("─" * 50)
    seed_all(args.seed)
    snn_model = SmallSNN(in_channels=3, n_classes=10, T=args.T).to(device)
    snn_tracker = run_snn(snn_model, train_loader, val_loader, probe_images,
                          device, epochs=args.epochs, lr=args.lr)
    snn_tracker.save(os.path.join(args.out_dir, "snn_repr_drift.json"))

    # -----------------------------------------------------------------------
    # Combined summary
    # -----------------------------------------------------------------------
    summary = build_summary(ann_tracker, snn_tracker)
    summary_path = os.path.join(args.out_dir, "repr_drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Summary saved → {summary_path}")

    # -----------------------------------------------------------------------
    # Print key findings
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  KEY FINDINGS")
    print("=" * 65)

    for algo, d in summary.items():
        corr = d["sparsity_drift_correlation"]
        print(f"\n  {algo}")
        print(f"    Final test accuracy:      {d['test_acc'][-1]:.1f}%")
        print(f"    Mean CKA drift (final 5): {np.mean(d['mean_cka_drift'][-5:]):.5f}")
        print(f"    Mean L2  drift (final 5): {np.mean(d['mean_l2_drift'][-5:]):.3f}")
        print(f"    Mean sparsity:            {np.mean(d['mean_sparsity']):.3f}")
        print(f"    Sparsity-drift corr:      r={corr['pearson_r']:.3f}")
        print(f"    Interpretation:           {corr['interpretation']}")

    ann_cka  = np.mean(summary['ANN']['mean_cka_drift'][-5:])
    snn_cka  = np.mean(summary['SNN_TET']['mean_cka_drift'][-5:])
    print("\n" + "─" * 50)
    if snn_cka > ann_cka * 1.1:
        print("  → SNN drifts MORE than ANN (surrogate gradient instability)")
    elif snn_cka < ann_cka * 0.9:
        print("  → SNN drifts LESS than ANN (sparsity stabilises representations)")
    else:
        print("  → ANN and SNN drift comparably (surrogate gradient error is small)")
    print("─" * 50)
    print("\nRun `python plot_task2.py` to generate figures.")


if __name__ == "__main__":
    main()
