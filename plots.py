import numpy as np
import matplotlib.pyplot as plt
import os

def plot_calibration(probs, targets, save_path="plots/calibration.png", dataset_name="Data"):
    """
    Reliability diagram: confidence vs the actual accuracy.
    Manually stores the confidence scores to safely handle multiclass softmax probabilities.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Model outputs 10 probabilities per image; grabs highest
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets)

    bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_confidences = []

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
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram - {dataset_name}")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_acc_vs_confidence(probs, targets, save_path="plots/acc_vs_conf.png", dataset_name="Data"):
    """Bucket predictions by confidence, plot accuracy per bucket as a bar chart."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, 11)
    accs = []
    bin_centers = []

    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            accs.append((preds[mask] == targets[mask]).mean())
            bin_centers.append((low + high) / 2)

    plt.figure(figsize=(7, 5))
    plt.bar(bin_centers, accs, width=0.08, edgecolor="black", alpha=0.7, label="Model Accuracy")
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration")
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence (Max Softmax Probability)")
    plt.ylabel("Actual Accuracy")
    plt.title(f"Accuracy vs Confidence - {dataset_name}")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()