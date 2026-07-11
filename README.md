# Representation Drift in Spiking Neural Networks
### IIT Madras Research Internship

## Overview
This project investigates representation drift in Spiking Neural Networks (SNNs) under surrogate gradient training. We study whether surrogate gradient approximations cause SNN representations to drift more than equivalent Artificial Neural Networks (ANNs), and whether the sparsity of spiking activity helps counteract that drift.

---

## Repository Structure

    snn-drift-analysis/
    task1/
        snn_model.py              - Shared SNN architecture (LIF neurons)
        drift_tracker.py          - Weight drift + firing rate drift tracker
        train_tet.py              - Temporal Efficient Training loop
        train_ebp.py              - Event-Based Backprop training loop
        main_experiment.py        - Runs both algorithms, saves results
        plot_drift.py             - Generates comparison figures
        results/
            drift_summary.json
            tet_drift.json
            ebp_drift.json
            figures/
    task2/
        ann_model.py              - ANN baseline (ReLU, identical architecture)
        snn_model.py              - SNN with TET surrogate gradient
        representation_drift_tracker.py
        train_models.py
        main_task2.py
        plot_task2.py
        results/
            figures/

---

## Task 1: Drift Analysis — TET vs Event-Based Backprop

### What we measured
- Weight drift: ||W_t - W_{t-1}||_2 per layer per epoch
- Firing rate drift: |r_t - r_{t-1}| per LIF layer per epoch

### Key Findings
- TET reached 85.6% accuracy on CIFAR-10 with smooth, stable weight drift decay
- EBP collapsed to 8.1% accuracy due to the silent neuron problem
- TET weight drift decayed smoothly from 4.4 to 0.009 over 30 epochs
- EBP weight drift froze to exactly 0.0 after epoch 20, confirming complete training collapse

### Run
    cd task1
    python main_experiment.py --epochs 30 --T 4 --batch_size 128
    python plot_drift.py

---

## Task 2: Representation Drift — ANN vs SNN

### What we measured
- CKA Drift: 1 - CKA(H_t, H_{t-1}) per layer per epoch
- L2 Drift: mean ||h_t(x) - h_{t-1}(x)||_2 across probe images
- Sparsity: fraction of silent activations per layer
- Sparsity-Drift Correlation: Pearson r between sparsity and CKA drift

### Research Questions
1. Do surrogate gradients cause SNN representations to drift more than ANNs?
2. Does spiking sparsity help counteract representation drift?

### Run
    cd task2
    python main_task2.py --epochs 30 --T 4 --batch_size 128
    python plot_task2.py

---

## Architecture
Both tasks use SmallSNN — a minimal 3-block convolutional SNN:
- 3 x ConvLIF blocks (Conv2d + BatchNorm + LIF neuron)
- T=4 simulation timesteps
- Task 2 ANN baseline replaces LIF with ReLU, keeping all else identical
- Dataset: CIFAR-10 (auto-downloaded on first run)

---

## Setup
    pip install torch torchvision matplotlib numpy

---

IIT Madras Research Internship — Brain & Intelligence Lab
