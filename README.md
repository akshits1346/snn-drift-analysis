# Representation Drift in Spiking Neural Networks
### IIT Madras Research Internship — Brain & Intelligence Lab

## Overview
This project investigates representation drift in Spiking Neural Networks (SNNs) under surrogate gradient training. We study whether surrogate gradient approximations cause SNN representations to drift more than equivalent Artificial Neural Networks (ANNs), and whether the sparsity of spiking activity helps counteract that drift.

---

## Repository Structure

    snn-drift-analysis/
    task1/
        snn_model.py                - Shared SNN architecture (LIF neurons)
        drift_tracker.py            - Weight drift + firing rate drift tracker
        train_tet.py                - Temporal Efficient Training loop
        train_ebp.py                - Event-Based Backprop training loop (original)
        train_ebp_fixed.py          - Event-Based Backprop (fixed — 3 targeted patches)
        main_experiment.py          - Runs TET vs EBP, saves results
        main_experiment_v2.py       - Runs TET vs EBP vs EBP-fixed for comparison
        plot_drift.py               - Generates comparison figures
        plot_drift_v2.py            - Generates fix comparison figure
        results/
            drift_summary.json      - TET vs EBP results
            drift_summary_v2.json   - TET vs EBP vs EBP-fixed results
            tet_drift.json
            ebp_drift.json
            ebp_original_drift.json
            ebp_fixed_drift.json
            figures/
    task2/
        ann_model.py                - ANN baseline (ReLU, identical architecture)
        snn_model.py                - SNN with TET surrogate gradient
        representation_drift_tracker.py
        train_models.py
        main_task2.py
        plot_task2.py
        results/
            repr_drift_summary.json
            ann_repr_drift.json
            snn_repr_drift.json
            figures/

---

## Task 1: Drift Analysis — TET vs Event-Based Backprop

### What we measured
- Weight drift: ||W_t - W_{t-1}||_2 per layer per epoch
- Firing rate drift: |r_t - r_{t-1}| per LIF layer per epoch

### Key Findings
- TET reached 85.6% accuracy on CIFAR-10 with smooth, stable weight drift decay
- TET weight drift decayed monotonically from 4.391 to 0.009 over 30 epochs (475x reduction)
- EBP collapsed to 8.1% accuracy (random guessing) due to the silent neuron problem
- EBP weight drift froze to exactly 0.0 after epoch 20 — the network completely stopped learning

### EBP Failure: Diagnosis
EBP's count+ loss only sends gradient to neurons that already spiked. At random initialisation with threshold=1.0, many neurons never fire, receive zero gradient, and remain permanently silent. This is a self-reinforcing cycle: silence → no gradient → no update → continued silence.

### EBP Fix: Three Targeted Interventions
1. Lower LIF threshold (1.0 → 0.5) — neurons fire more easily at random init
2. TET warmup for 5 epochs — establishes healthy firing before switching to EBP
3. Tighter gradient clipping (1000 → 1.0) — prevents exploding gradients killing neurons

### EBP Fix Results
- Fixed EBP reached 28% accuracy (up from 8.1%), confirming the diagnosis
- Accuracy crashes from 71% to 23% when switching from warmup to EBP at epoch 6
- Suggests count+ loss hyperparameters are too aggressive for this architecture

### Run
    cd task1
    python main_experiment.py --epochs 30 --T 4 --batch_size 128
    python main_experiment_v2.py --epochs 30 --T 4 --batch_size 128
    python plot_drift.py
    python plot_drift_v2.py

---

## Task 2: Representation Drift — ANN vs SNN

### What we measured
- CKA Drift: 1 - CKA(H_t, H_{t-1}) per layer per epoch
- L2 Drift: mean ||h_t(x) - h_{t-1}(x)||_2 across 512 fixed probe images
- Sparsity: fraction of silent activations per layer
- Sparsity-Drift Correlation: Pearson r between sparsity and CKA drift

### Research Questions
1. Do surrogate gradients cause SNN representations to drift more than ANNs?
2. Does spiking sparsity help counteract representation drift?

### Key Findings
- Both ANN and SNN reached equivalent final accuracy (~86%), confirming a fair comparison
- SNN drifts 2.14x more than ANN by CKA metric (0.00114 vs 0.00053, final 5 epochs)
- SNN drifts 2.85x more than ANN by L2 metric (10.55 vs 3.70, final 5 epochs)
- SNN sparsity is 86.3% vs ANN sparsity of 60.0% — SNN is dramatically more sparse
- Sparsity strongly counteracts drift in ANN (r = -0.521)
- Sparsity does NOT counteract drift in SNN (r = -0.017)
- Conclusion: surrogate gradient noise completely overwhelms the stabilising effect of sparsity in SNNs

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
- ~1.5M parameters

---

## Setup
    pip install torch torchvision matplotlib numpy

---

*IIT Madras Research Internship — Brain & Intelligence Lab | July 2026*
