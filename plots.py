import numpy as np
import matplotlib.pyplot as plt
import os

def plot_calibration(probs, targets, save_path="plots/calibration.png", dataset_name="Data"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets)

    bins = np.linspace(0, 1, 11)
    bin_accuracies, bin_confidences = [], []

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            bin_accuracies.append(correct[mask].mean())
            bin_confidences.append(confidences[mask].mean())
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append((low + high) / 2)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(bin_confidences, bin_accuracies, marker="o", color="blue", label=f"Model ({dataset_name})")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram - {dataset_name}")
    plt.legend(loc="upper left"); plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_acc_vs_confidence(probs, targets, save_path="plots/acc_vs_conf.png", dataset_name="Data"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, 11)
    accs, bin_centers = [], []

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            accs.append((preds[mask] == targets[mask]).mean())
            bin_centers.append((low + high) / 2)

    plt.figure(figsize=(7, 5))
    plt.bar(bin_centers, accs, width=0.08, edgecolor="black", alpha=0.7, label="Model Accuracy")
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Confidence (Max Softmax Probability)"); plt.ylabel("Actual Accuracy")
    plt.title(f"Accuracy vs Confidence - {dataset_name}")
    plt.legend(loc="upper left"); plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_layer_comparison(layer_results, save_path="plots/ocsvm_layer_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    layers = [r["layer"] for r in layer_results]
    aurocs = [r["auroc"] for r in layer_results]

    plt.figure(figsize=(6, 4))
    plt.bar(layers, aurocs, color="teal", alpha=0.8, edgecolor="black")
    plt.ylabel("AUROC (Failure Detection)")
    plt.title("OCSVM Performance by Feature Layer")
    plt.ylim(0.4, 1.0)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_hyperparam_heatmap(sweep_results, save_path="plots/ocsvm_hyperparam_sweep.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    top_10 = sweep_results[:10]
    labels = [f"k:{r['kernel']}, nu:{r['nu']}, g:{r['gamma']}" for r in top_10]
    aurocs = [r["auroc"] for r in top_10]

    plt.figure(figsize=(10, 4))
    plt.barh(labels[::-1], aurocs[::-1], color="mediumpurple", edgecolor="black")
    plt.xlabel("AUROC")
    plt.title("Top 10 OCSVM Hyperparameter Configurations")
    plt.xlim(0.4, 1.0)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_auroc_summary(all_results, detector_names, save_path="plots/auroc_summary.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shifts = list(all_results.keys())
    x = np.arange(len(shifts))
    width = 0.11

    plt.figure(figsize=(12, 6))
    for i, d in enumerate(detector_names):
        scores = [all_results[s][d] for s in shifts]
        plt.bar(x + i * width, scores, width, label=d)

    plt.xlabel("Distribution Shift")
    plt.ylabel("AUROC")
    plt.title("Failure Detection Performance Across Shifts")
    plt.xticks(x + width * (len(detector_names) - 1) / 2, shifts, rotation=15)
    plt.ylim(0.4, 1.0)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_severity_sweep(severity_results, save_path="plots/severity_sweep.png"):
    """Plots Model Accuracy drop alongside OOD Detection AUROC across severity levels (1 to 5)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    severities = [1, 2, 3, 4, 5]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    for shift_type, data in severity_results.items():
        accs = [data[sev]["accuracy"] for sev in severities]
        aurocs = [data[sev]["ood_auroc"] for sev in severities]

        ax1.plot(severities, accs, linestyle="--", label=f"{shift_type} Accuracy")
        ax1.plot(severities, aurocs, linestyle="-", label=f"{shift_type} OOD AUROC")

    ax1.set_xlabel("Shift Severity Level (1-5)")
    ax1.set_ylabel("Metric Score")
    ax1.set_title("Practical Monitoring: Accuracy Drop vs OOD Detection Performance")
    ax1.set_xticks(severities)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
