"""
plot_drift.py
=============
Reads results/drift_summary.json and produces 4 publication-quality figures:

  1. Test accuracy vs epoch          (TET vs EBP)
  2. Mean weight drift vs epoch      (TET vs EBP)
  3. Mean firing-rate drift vs epoch (TET vs EBP)
  4. Per-layer weight drift          (heatmap, one panel per algo)

All figures are saved to results/figures/.

Usage:
    python plot_drift.py [--summary path/to/drift_summary.json]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {"TET": "#E05A5A", "EBP": "#5A90E0"}
LINESTYLES = {"TET": "-", "EBP": "--"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "font.size": 11,
})


def load(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fig 1: Accuracy + Loss
# ---------------------------------------------------------------------------
def plot_accuracy_loss(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for algo, d in data.items():
        epochs = d["epochs"]
        axes[0].plot(epochs, d["test_acc"],
                     color=COLORS[algo], ls=LINESTYLES[algo], label=algo)
        axes[1].plot(epochs, d["loss"],
                     color=COLORS[algo], ls=LINESTYLES[algo], label=algo)

    axes[0].set(title="Test Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    for ax in axes:
        ax.legend()

    fig.suptitle("TET vs EBP — Accuracy & Loss", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_accuracy_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Fig 2: Weight drift over training
# ---------------------------------------------------------------------------
def plot_weight_drift(data, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))

    for algo, d in data.items():
        epochs = d["epochs"]
        ax.plot(epochs, d["mean_weight_drift"],
                color=COLORS[algo], ls=LINESTYLES[algo], label=algo)

    ax.set(
        title="Mean Weight Drift Over Training\n"
              r"$\|W_t - W_{t-1}\|_2$ (averaged across layers)",
        xlabel="Epoch",
        ylabel=r"$\Delta W$ (L2 norm)",
    )
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_weight_drift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Fig 3: Firing-rate drift over training
# ---------------------------------------------------------------------------
def plot_firing_rate_drift(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for algo, d in data.items():
        epochs = d["epochs"]
        axes[0].plot(epochs, d["mean_fire_rate_drift"],
                     color=COLORS[algo], ls=LINESTYLES[algo], label=algo)

    axes[0].set(
        title="Mean Firing-Rate Drift Over Training\n"
              r"$|\bar{r}_t - \bar{r}_{t-1}|$ (averaged across LIF layers)",
        xlabel="Epoch",
        ylabel=r"$\Delta \bar{r}$",
    )
    axes[0].legend()

    # Also show absolute firing rate (both algos, one layer each)
    for algo, d in data.items():
        layers = list(d["fire_rates_layers"].keys())
        if layers:
            name = layers[0]
            rates = d["fire_rates_layers"][name]
            epochs = d["epochs"][:len(rates)]
            axes[1].plot(epochs, rates,
                         color=COLORS[algo], ls=LINESTYLES[algo],
                         label=f"{algo} ({name})")

    axes[1].set(
        title="Absolute Firing Rate — First LIF Layer",
        xlabel="Epoch",
        ylabel="Mean spike probability",
    )
    axes[1].legend()

    fig.suptitle("TET vs EBP — Firing-Rate Drift", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_firing_rate_drift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Fig 4: Per-layer weight drift heatmap
# ---------------------------------------------------------------------------
def plot_layer_heatmap(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, (algo, d) in zip(axes, data.items()):
        layer_drift = d["weight_drift_layers"]
        if not layer_drift:
            ax.set_title(f"{algo} — no layer data")
            continue

        layer_names = list(layer_drift.keys())
        matrix = np.array([layer_drift[k] for k in layer_names])  # [layers, epochs]

        epochs = np.arange(1, matrix.shape[1] + 1)
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            cmap="YlOrRd",
            extent=[epochs[0] - 0.5, epochs[-1] + 0.5, len(layer_names) - 0.5, -0.5],
        )
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_title(f"{algo} — Per-Layer Weight Drift\n"
                     r"$\|W_t - W_{t-1}\|_2$")
        fig.colorbar(im, ax=ax, label=r"$\Delta W$")

    fig.suptitle("Per-Layer Weight Drift Heatmap", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_layer_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Fig 5: Combined 4-panel overview (good for a report/presentation)
# ---------------------------------------------------------------------------
def plot_overview(data, out_dir):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_wd   = fig.add_subplot(gs[1, 0])
    ax_frd  = fig.add_subplot(gs[1, 1])

    for algo, d in data.items():
        ep = d["epochs"]
        kw = dict(color=COLORS[algo], ls=LINESTYLES[algo], label=algo)
        ax_acc.plot(ep, d["test_acc"],            **kw)
        ax_loss.plot(ep, d["loss"],               **kw)
        ax_wd.plot(ep, d["mean_weight_drift"],    **kw)
        ax_frd.plot(ep, d["mean_fire_rate_drift"],**kw)

    ax_acc.set(title="Test Accuracy (%)",         xlabel="Epoch")
    ax_loss.set(title="Training Loss",            xlabel="Epoch")
    ax_wd.set(title=r"Mean Weight Drift $\|\Delta W\|_2$", xlabel="Epoch")
    ax_frd.set(title=r"Mean Firing-Rate Drift $|\Delta \bar{r}|$", xlabel="Epoch")

    for ax in [ax_acc, ax_loss, ax_wd, ax_frd]:
        ax.legend(fontsize=9)

    fig.suptitle("SNN Drift Analysis: TET vs Event-Based Backprop\n"
                 "(CIFAR-10, SmallSNN)", fontsize=14, fontweight="bold")
    path = os.path.join(out_dir, "fig5_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="./results/drift_summary.json")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(args.summary), "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.summary} ...")
    data = load(args.summary)

    print("Generating figures ...")
    plot_accuracy_loss(data, out_dir)
    plot_weight_drift(data, out_dir)
    plot_firing_rate_drift(data, out_dir)
    plot_layer_heatmap(data, out_dir)
    plot_overview(data, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
