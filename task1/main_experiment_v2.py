"""
main_experiment_v2.py
=====================
Runs TET + EBP (original, broken) + EBP (fixed) for comparison.
Shows clearly what the bug was and that the fix works.

Usage:
    python main_experiment_v2.py --epochs 30 --T 4 --batch_size 128
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from snn_model import SmallSNN
from train_tet import run_tet
from train_ebp import run_ebp
from train_ebp_fixed import run_ebp as run_ebp_fixed
from drift_tracker import DriftTracker


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--T",          type=int,   default=4)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--warmup",     type=int,   default=5)
    parser.add_argument("--data_root",  type=str,   default="./data")
    parser.add_argument("--out_dir",    type=str,   default="./results_v2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  SNN DRIFT: TET vs EBP (original) vs EBP (fixed)")
    print(f"  epochs={args.epochs}, T={args.T}, warmup={args.warmup}")
    print("=" * 65)
    print(f"Device: {device}\n")

    train_loader, val_loader = get_cifar10(args.batch_size, args.data_root)

    # --- TET ---
    print("\n--- PHASE 1: TET ---")
    seed_all(args.seed)
    m1 = SmallSNN(3, 10, args.T).to(device)
    t1 = run_tet(m1, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
    t1.save(os.path.join(args.out_dir, "tet_drift.json"))

    # --- EBP original ---
    print("\n--- PHASE 2: EBP (original, broken) ---")
    seed_all(args.seed)
    m2 = SmallSNN(3, 10, args.T).to(device)
    t2 = run_ebp(m2, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
    t2.save(os.path.join(args.out_dir, "ebp_original_drift.json"))

    # --- EBP fixed ---
    print("\n--- PHASE 3: EBP (fixed) ---")
    seed_all(args.seed)
    m3 = SmallSNN(3, 10, args.T).to(device)
    t3 = run_ebp_fixed(m3, train_loader, val_loader, device,
                       epochs=args.epochs, lr=args.lr, warmup_epochs=args.warmup)
    t3.save(os.path.join(args.out_dir, "ebp_fixed_drift.json"))

    # --- Summary ---
    summary = {}
    for tracker in [t1, t2, t3]:
        summary[tracker.algo] = {
            "epochs":            [e["epoch"]            for e in tracker.epoch_log],
            "test_acc":          [e["test_acc"]          for e in tracker.epoch_log],
            "mean_weight_drift": [e["mean_weight_drift"] for e in tracker.epoch_log],
            "mean_fire_rate_drift": [e["mean_fire_rate_drift"] for e in tracker.epoch_log],
        }

    with open(os.path.join(args.out_dir, "drift_summary_v2.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    for algo, d in summary.items():
        print(f"  {algo:12s} | test_acc={d['test_acc'][-1]:.1f}% | "
              f"final_Δw={d['mean_weight_drift'][-1]:.5f}")

    print("\nRun `python plot_drift_v2.py` to generate figures.")


if __name__ == "__main__":
    main()
