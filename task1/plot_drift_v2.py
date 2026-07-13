"""
plot_drift_v2.py
================
Plots TET vs EBP (original) vs EBP (fixed) side by side.
Clearly shows the diagnosis and the fix working.

Usage:
    python plot_drift_v2.py [--summary ./results_v2/drift_summary_v2.json]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "TET":       "#E05A5A",
    "EBP":       "#5A90E0",
    "EBP_fixed": "#2CA02C",
}
LABELS = {
    "TET":       "TET",
    "EBP":       "EBP (original — collapsed)",
    "EBP_fixed": "EBP (fixed — warmup + threshold + clipping)",
}
LS = {"TET": "-", "EBP": "--", "EBP_fixed": "-."}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.2,
    "font.size": 11,
})


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="./results_v2/drift_summary_v2.json")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(args.summary), "figures")
    os.makedirs(out_dir, exist_ok=True)

    data = load(args.summary)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for algo, d in data.items():
        kw = dict(color=COLORS.get(algo, "gray"),
                  ls=LS.get(algo, "-"),
                  label=LABELS.get(algo, algo))
        ep = d["epochs"]
        axes[0].plot(ep, d["test_acc"],             **kw)
        axes[1].plot(ep, d["mean_weight_drift"],    **kw)
        axes[2].plot(ep, d["mean_fire_rate_drift"], **kw)

    axes[0].set(title="Test Accuracy (%)",
                xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].set(title="Mean Weight Drift ||ΔW||₂",
                xlabel="Epoch", ylabel="L2 norm")
    axes[2].set(title="Mean Firing-Rate Drift |Δr̄|",
                xlabel="Epoch", ylabel="Δr̄")

    for ax in axes:
        ax.legend(fontsize=8)

    # Annotate the fix
    axes[0].annotate("EBP fixed →\nlearning again",
                     xy=(10, data.get("EBP_fixed", {}).get("test_acc", [20]*15)[14]),
                     xytext=(15, 40),
                     arrowprops=dict(arrowstyle="->", color="green"),
                     color="green", fontsize=9)

    fig.suptitle("EBP Diagnosis & Fix: Silent Neuron Problem\n"
                 "Fixes: LIF threshold 1.0→0.5 | TET warmup (5 epochs) | grad clip 1000→1.0",
                 fontweight="bold", fontsize=12)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_ebp_fix_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
