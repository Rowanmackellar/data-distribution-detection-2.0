# Data Distribution & Model Failure Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

A research pipeline for evaluating **model failure detection under distribution shift**.

The project trains a **ResNet-18** on CIFAR-10 and evaluates **seven uncertainty/OOD detection methods** on six common image corruptions, each evaluated across five severity levels. Instead of only detecting out-of-distribution samples, every method is measured by its ability to predict **when the classifier will make an incorrect prediction**.

---

# Features

- Train a ResNet-18 classifier on CIFAR-10
- Generate six distribution shifts
- Evaluate seven failure detection methods
- Compute AUROC for predicting model failures
- Automatically tune the One-Class SVM detector
- Generate calibration, confidence, and comparison plots
- Save all metrics for reproducibility

---

# Detection Methods

The repository evaluates seven complementary approaches.

| Method | Description |
|---------|-------------|
| Confidence | Softmax entropy from a single forward pass |
| Temperature Scaling | Entropy after logit calibration |
| Energy Score | Free-energy score |
| Embedding Distance | Mean k-NN distance in feature space |
| Monte Carlo Dropout | Predictive entropy from multiple stochastic passes |
| One-Class SVM | Unsupervised anomaly detector trained on clean embeddings |
| Mahalanobis Distance | Class-conditional Gaussian distance |

Each detector produces a score estimating how likely a prediction is to be incorrect. Performance is reported using **AUROC** against the classifier's actual errors.

---

# Repository Structure

```
.
├── train.py                 # Train ResNet-18
├── evaluate.py              # Evaluate all detectors
├── detectors.py             # Detection algorithms
├── plots.py                 # Visualization utilities
├── models/                  # Saved checkpoints
├── plots/                   # Generated figures
└── results/
    └── evaluation_results.json
```

---

# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- pip

Clone the repository

```bash
git clone https://github.com/Rowanmackellar/data-distribution-detection-2.0.git
cd data-distribution-detection-2.0
```

Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

## 1. Train the classifier

```bash
python train.py
```

This trains a ResNet-18 on CIFAR-10 and saves the model checkpoint.

---

## 2. Evaluate failure detectors

```bash
python evaluate.py
```

The evaluation script will

- load the trained model
- generate corrupted test datasets
- evaluate every detection method
- compute AUROC for every corruption
- automatically tune the One-Class SVM
- save all numerical results

Results are written to

```
results/evaluation_results.json
```

---

## 3. Generate visualizations

If plots are generated separately,

```bash
python plots.py
```

Otherwise, `evaluate.py` generates them automatically.

---

# One-Class SVM Tuning

Because the One-Class SVM is highly sensitive to feature representation and hyperparameters, the evaluation pipeline performs two tuning stages before the final evaluation.

### Embedding Layer Comparison

Feature embeddings are extracted from

- layer2
- layer3
- final pooled features

The layer with the highest validation AUROC is selected.

Output:

```
plots/ocsvm_layer_comparison.png
```

---

### Hyperparameter Search

The best embedding layer is then used for a grid search over

- kernel
- gamma
- nu

The best-performing configuration is used for the reported results.

Output:

```
plots/ocsvm_hyperparam_sweep.png
```

---

# Results

Running

```bash
python evaluate.py
```

produces a table similar to

| Shift | Confidence | Temp | Energy | Distance | MC Dropout | OCSVM | Mahalanobis |
|--------|-----------:|-----:|--------:|----------:|-----------:|------:|-------------:|
| Blur | | | | | | | |
| Noise | | | | | | | |
| Brightness / Contrast | | | | | | | |
| Rotation | | | | | | | |
| Occlusion | | | | | | | |
| JPEG | | | | | | | |

The exact numbers are automatically generated from the current model and written to

```
results/evaluation_results.json
```

---

# Generated Figures

The pipeline automatically creates figures under `plots/`.

| Figure | Description |
|---------|-------------|
| calibration_clean.png | Reliability diagram on clean data |
| calibration_<shift>.png | Reliability diagrams for each corruption |
| acc_vs_conf_clean.png | Accuracy vs confidence on clean data |
| acc_vs_conf_<shift>.png | Accuracy vs confidence under shift |
| ocsvm_layer_comparison.png | Embedding layer comparison |
| ocsvm_hyperparam_sweep.png | Hyperparameter search visualization |
| auroc_summary.png | AUROC comparison across all methods |
| confident_failures.png | Highest-confidence incorrect predictions |

---

# Evaluation Metric

The primary metric is **Area Under the Receiver Operating Characteristic (AUROC)**.

- **1.0** — perfect failure prediction
- **0.5** — random guessing

Higher AUROC indicates a detector is better at identifying predictions that will be incorrect.
