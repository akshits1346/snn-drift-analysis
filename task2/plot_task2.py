"""
plot_task2.py
=============
Generates all figures for Task 2 representation drift analysis.

Figures:
  fig1 — Accuracy curves (ANN vs SNN)
  fig2 — CKA drift over training (mean + per layer)
  fig3 — L2 representation drift over training
  fig4 — Sparsity over training (SNN vs ANN)
  fig5 — Sparsity vs CKA drift scatter (the key correlation plot)
  fig6 — Combined overview (6 panels, report-ready)

Usage:
    python plot_task2.py [--summary ./results_task2/repr_drift_summary.json]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

COLORS = {"ANN": "#3A86FF", "SNN_TET": "#FF006E"}
LABELS = {"ANN": "ANN (ReLU)", "SNN_TET": "SNN (TET / Surrogate Grad)"}
LS     = {"ANN": "-", "SNN_TET": "--"}

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


# ---------------------------------------------------------------------------
# Fig 1: Accuracy
# ---------------------------------------------------------------------------
def plot_accuracy(data, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for algo, d in data.items():
        ax.plot(d["epochs"], d["test_acc"],
                color=COLORS[algo], ls=LS[algo], label=LABELS[algo])
    ax.set(title="Test Accuracy: ANN vs SNN", xlabel="Epoch", ylabel="Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_accuracy.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


# ---------------------------------------------------------------------------
# Fig 2: CKA drift (mean + per layer)
# ---------------------------------------------------------------------------
def plot_cka_drift(data, out_dir):
    layers = list(list(data.values())[0]["cka_drift_layers"].keys())
    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(5 * (len(layers) + 1), 4))

    # Mean
    ax = axes[0]
    for algo, d in data.items():
        ax.plot(d["epochs"], d["mean_cka_drift"],
                color=COLORS[algo], ls=LS[algo], label=LABELS[algo])
    ax.set(title="Mean CKA Drift\n(all layers)", xlabel="Epoch",
           ylabel="1 − CKA(H_t, H_{t−1})")
    ax.legend(fontsize=9)

    # Per layer
    for i, layer in enumerate(layers):
        ax = axes[i + 1]
        for algo, d in data.items():
            vals = d["cka_drift_layers"].get(layer, [])
            ax.plot(d["epochs"][:len(vals)], vals,
                    color=COLORS[algo], ls=LS[algo], label=LABELS[algo])
        ax.set(title=f"CKA Drift — {layer}", xlabel="Epoch")
        ax.legend(fontsize=8)

    fig.suptitle("Representation Drift (CKA): ANN vs SNN", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig2_cka_drift.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


# ---------------------------------------------------------------------------
# Fig 3: L2 drift
# ---------------------------------------------------------------------------
def plot_l2_drift(data, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for algo, d in data.items():
        ax.plot(d["epochs"], d["mean_l2_drift"],
                color=COLORS[algo], ls=LS[algo], label=LABELS[algo])
    ax.set(
        title="Mean L2 Representation Drift\n"
              r"$\mathbb{E}_x[\|h_t(x) - h_{t-1}(x)\|_2]$",
        xlabel="Epoch", ylabel="Mean L2 distance"
    )
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "fig3_l2_drift.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


# ---------------------------------------------------------------------------
# Fig 4: Sparsity over training
# ---------------------------------------------------------------------------
def plot_sparsity(data, out_dir):
    layers = list(list(data.values())[0]["sparsity_layers"].keys())
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))
    if len(layers) == 1:
        axes = [axes]

    for i, layer in enumerate(layers):
        ax = axes[i]
        for algo, d in data.items():
            vals = d["sparsity_layers"].get(layer, [])
            ax.plot(d["epochs"][:len(vals)], vals,
                    color=COLORS[algo], ls=LS[algo], label=LABELS[algo])
        ax.set(title=f"Sparsity — {layer}\n(fraction of silent activations)",
               xlabel="Epoch", ylabel="Sparsity")
        ax.legend(fontsize=8)

    fig.suptitle("Activation Sparsity: ANN (ReLU zeros) vs SNN (non-spiking neurons)",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig4_sparsity.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


# ---------------------------------------------------------------------------
# Fig 5: Sparsity vs CKA drift scatter — THE KEY PLOT
# ---------------------------------------------------------------------------
def plot_sparsity_vs_drift(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (algo, d) in zip(axes, data.items()):
        layers = list(d["cka_drift_layers"].keys())
        colors_l = plt.cm.viridis(np.linspace(0.2, 0.8, len(layers)))

        for layer, col in zip(layers, colors_l):
            sp = d["sparsity_layers"].get(layer, [])
            cd = d["cka_drift_layers"].get(layer, [])
            n = min(len(sp), len(cd))
            ax.scatter(sp[:n], cd[:n], color=col, alpha=0.7, s=30, label=layer)

        # overall regression line
        all_sp = [v for layer in layers for v in d["sparsity_layers"].get(layer, [])]
        all_cd = [v for layer in layers for v in d["cka_drift_layers"].get(layer, [])]
        n = min(len(all_sp), len(all_cd))
        if n > 2:
            z = np.polyfit(all_sp[:n], all_cd[:n], 1)
            xr = np.linspace(min(all_sp[:n]), max(all_sp[:n]), 100)
            ax.plot(xr, np.polyval(z, xr), 'k--', lw=1.5, label='trend')

        corr = d["sparsity_drift_correlation"]
        r = corr["pearson_r"]
        ax.set(
            title=f"{LABELS[algo]}\nSparsity vs CKA Drift  (r={r:.3f})",
            xlabel="Sparsity (fraction silent)",
            ylabel="CKA Drift (1 − CKA)"
        )
        ax.legend(fontsize=8)
        ax.text(0.05, 0.93, corr["interpretation"],
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("Does Sparsity Counteract Representation Drift?", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig5_sparsity_vs_drift.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


# ---------------------------------------------------------------------------
# Fig 6: Overview (report-ready, 6 panels)
# ---------------------------------------------------------------------------
def plot_overview(data, out_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_cka  = fig.add_subplot(gs[0, 1])
    ax_l2   = fig.add_subplot(gs[0, 2])
    ax_sp   = fig.add_subplot(gs[1, 0])
    ax_scat = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")

    for algo, d in data.items():
        kw = dict(color=COLORS[algo], ls=LS[algo], label=LABELS[algo])
        ep = d["epochs"]
        ax_acc.plot(ep, d["test_acc"],       **kw)
        ax_cka.plot(ep, d["mean_cka_drift"], **kw)
        ax_l2.plot( ep, d["mean_l2_drift"],  **kw)
        ax_sp.plot( ep, d["mean_sparsity"],  **kw)

        # scatter (all layers combined)
        layers = list(d["cka_drift_layers"].keys())
        all_sp = [v for l in layers for v in d["sparsity_layers"].get(l, [])]
        all_cd = [v for l in layers for v in d["cka_drift_layers"].get(l, [])]
        n = min(len(all_sp), len(all_cd))
        ax_scat.scatter(all_sp[:n], all_cd[:n],
                        color=COLORS[algo], alpha=0.4, s=15, label=LABELS[algo])

    ax_acc.set(title="Test Accuracy (%)",          xlabel="Epoch")
    ax_cka.set(title="CKA Drift (1−CKA)",          xlabel="Epoch")
    ax_l2.set( title="L2 Repr Drift",              xlabel="Epoch")
    ax_sp.set( title="Mean Sparsity",              xlabel="Epoch")
    ax_scat.set(title="Sparsity vs CKA Drift",
                xlabel="Sparsity", ylabel="CKA Drift")

    for ax in [ax_acc, ax_cka, ax_l2, ax_sp, ax_scat]:
        ax.legend(fontsize=8)

    # Key findings text box
    lines = ["KEY FINDINGS\n"]
    for algo, d in data.items():
        corr = d["sparsity_drift_correlation"]
        lines.append(f"{LABELS[algo]}:")
        lines.append(f"  Accuracy:  {d['test_acc'][-1]:.1f}%")
        lines.append(f"  CKA drift: {np.mean(d['mean_cka_drift'][-5:]):.5f}")
        lines.append(f"  Sparsity:  {np.mean(d['mean_sparsity']):.3f}")
        lines.append(f"  r(sp,Δ):   {corr['pearson_r']:.3f}\n")

    ann_cka = np.mean(data['ANN']['mean_cka_drift'][-5:])
    snn_cka = np.mean(data['SNN_TET']['mean_cka_drift'][-5:])
    if snn_cka > ann_cka * 1.1:
        verdict = "SNN drifts MORE\n(surrogate grad instability)"
    elif snn_cka < ann_cka * 0.9:
        verdict = "SNN drifts LESS\n(sparsity stabilises repr)"
    else:
        verdict = "ANN ≈ SNN drift\n(surrogate error is small)"
    lines.append(f"VERDICT:\n{verdict}")

    ax_text.text(0.05, 0.95, "\n".join(lines),
                 transform=ax_text.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle("Task 2: Representation Drift in SNNs under Surrogate Gradient Training\n"
                 "CIFAR-10 | SmallSNN (TET) vs SmallANN (ReLU)",
                 fontsize=13, fontweight="bold")

    p = os.path.join(out_dir, "fig6_overview.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="./results_task2/repr_drift_summary.json")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(args.summary), "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.summary} ...")
    data = load(args.summary)

    print("Generating figures ...")
    plot_accuracy(data, out_dir)
    plot_cka_drift(data, out_dir)
    plot_l2_drift(data, out_dir)
    plot_sparsity(data, out_dir)
    plot_sparsity_vs_drift(data, out_dir)
    plot_overview(data, out_dir)

    print(f"\nAll figures → {out_dir}/")


if __name__ == "__main__":
    main()
