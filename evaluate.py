import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from train import get_device
from src.models import ResNetCIFAR
from src.datasets import get_clean_dataloaders, get_shifted_dataloader
from detection_methods.detectors import (
    evaluate_confidence,
    evaluate_temperature_scaled,
    fit_temperature,
    evaluate_energy,
    evaluate_distance,
    evaluate_mc_dropout,
    train_ood_detector,
    evaluate_ood,
    compare_embedding_layers,
    sweep_ocsvm_hyperparameters,
    train_mahalanobis_detector,
    evaluate_mahalanobis,
)
from plots import plot_calibration, plot_acc_vs_confidence

# All distribution shifts evaluated, including the additional shifts requested
# in review: brightness/contrast, rotation, occlusion, and JPEG compression.
SHIFT_TYPES = ["blur", "noise", "brightness_contrast", "rotation", "occlusion", "jpeg"]
DETECTOR_NAMES = ["confidence", "temp_scaled", "energy", "distance", "mc_dropout",
                  "ood", "mahalanobis"]


# Raw prediction data, disabled gradient tracking, and saves memory putting Dropout or Normalization into eval
@torch.no_grad()
def get_probabilities_and_targets(model, loader, device):
    """Helper to extract raw softmax probabilities and true targets for calibration plots."""
    model.eval()
    all_probs, all_targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_targets)


# 7 failure detectors -> AUROC score per shift
def run_evaluation(model, train_loader, ood_detector, mahalanobis_params, temperature,
                    shift_type, device):
    """Runs all failure detection evaluations over a specific distribution shift."""
    shifted_loader = get_shifted_dataloader(shift_type=shift_type)
    results = {}

    # CONFIDENCE — single forward pass, softmax entropy
    scores, preds, targets = evaluate_confidence(model, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["confidence"] = roc_auc_score(failures, scores)

    # TEMPERATURE-SCALED CONFIDENCE - entropy after calibrating logits on clean val data
    scores, preds, targets = evaluate_temperature_scaled(model, shifted_loader, device,
                                                          temperature=temperature)
    failures = (preds != targets).astype(int)
    results["temp_scaled"] = roc_auc_score(failures, scores)

    # ENERGY SCORE - free-energy baseline
    scores, preds, targets = evaluate_energy(model, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["energy"] = roc_auc_score(failures, scores)

    # DISTANCE — k-NN distance from clean training embeddings
    scores, preds, targets = evaluate_distance(model, train_loader, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["distance"] = roc_auc_score(failures, scores)

    # MONTE CARLO DROPOUT — entropy over 15 stochastic passes
    scores, preds, targets = evaluate_mc_dropout(model, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["mc_dropout"] = roc_auc_score(failures, scores)

    # OOD DETECTOR — One-Class SVM trained on clean, standardized embeddings
    scores, preds, targets = evaluate_ood(model, ood_detector, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["ood"] = roc_auc_score(failures, scores)

    # MAHALANOBIS DISTANCE - class-conditional Gaussian baseline
    scores, preds, targets = evaluate_mahalanobis(model, mahalanobis_params, shifted_loader, device)
    failures = (preds != targets).astype(int)
    results["mahalanobis"] = roc_auc_score(failures, scores)

    # VISUALIZATIONS — Generate calibration plots per shift
    print(f"Generating calibration charts for shift: {shift_type}...")
    shift_probs, shift_targets = get_probabilities_and_targets(model, shifted_loader, device)
    plot_calibration(shift_probs, shift_targets, save_path=f"plots/calibration_{shift_type}.png",
                      dataset_name=f"Shift: {shift_type}")
    plot_acc_vs_confidence(shift_probs, shift_targets, save_path=f"plots/acc_vs_conf_{shift_type}.png",
                            dataset_name=f"Shift: {shift_type}")

    return results


def print_results(all_results):
    col_w = 12

    header = f"{'Shift':<18}" + "".join(f"{d:>{col_w}}" for d in DETECTOR_NAMES)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for shift_type, results in all_results.items():
        row = f"{shift_type:<18}" + "".join(f"{results[d]:>{col_w}.3f}" for d in DETECTOR_NAMES)
        print(row)

    print("=" * len(header) + "\n")


def print_layer_comparison(layer_results):
    print("OCSVM AUROC by embedding layer (validated on the blur shift):")
    for r in layer_results:
        print(f"  {r['layer']:<10} AUROC = {r['auroc']:.3f}")
    print()


def print_hyperparam_sweep(sweep_results, top_n=10):
    print(f"OCSVM hyperparameter sweep — top {top_n} configs (validated on the blur shift):")
    print(f"  {'kernel':<8}{'nu':>8}{'gamma':>10}{'AUROC':>10}")
    for r in sweep_results[:top_n]:
        print(f"  {r['kernel']:<8}{r['nu']:>8.3f}{str(r['gamma']):>10}{r['auroc']:>10.3f}")
    print()


if __name__ == "__main__":
    device = get_device()

    model = ResNetCIFAR().to(device)
    model.load_state_dict(torch.load("models/checkpoints/resnet18_best.pt", map_location=device))
    model.eval()

    print("Loading clean data...")
    train_loader, clean_test_loader = get_clean_dataloaders()

    # Baseline visualization - for clean data
    print("Generating clean baseline calibration charts...")
    clean_probs, clean_targets = get_probabilities_and_targets(model, clean_test_loader, device)
    plot_calibration(clean_probs, clean_targets, save_path="plots/calibration_clean.png",
                      dataset_name="Clean Baseline")
    plot_acc_vs_confidence(clean_probs, clean_targets, save_path="plots/acc_vs_conf_clean.png",
                            dataset_name="Clean Baseline")

    print("Fitting temperature scaling on clean held-out data...")
    temperature = fit_temperature(model, clean_test_loader, device)
    print(f"--> Learned temperature T = {temperature:.3f}")

    # Use the blur shift as a held-out validation shift for model-selection style
    # experiments (layer comparison, hyperparameter sweep) so the reported test
    # results across all shifts stay untouched by that tuning.
    print("Loading blur shift as validation set for OCSVM layer/hyperparameter tuning...")
    val_shifted_loader = get_shifted_dataloader(shift_type="blur")

    print("Comparing OCSVM AUROC across embedding layers (layer2, layer3, final)...")
    layer_results = compare_embedding_layers(model, train_loader, val_shifted_loader, device,
                                              layers=("layer2", "layer3", "final"))
    print_layer_comparison(layer_results)
    best_layer = layer_results[0]["layer"]
    print(f"--> Best-performing layer: {best_layer}")

    print(f"Sweeping OCSVM hyperparameters (nu, gamma, kernel) on layer '{best_layer}'...")
    sweep_results = sweep_ocsvm_hyperparameters(model, train_loader, val_shifted_loader, device,
                                                 layer=best_layer,
                                                 nus=(0.01, 0.05, 0.1, 0.2),
                                                 gammas=("scale", "auto"),
                                                 kernels=("rbf", "linear"))
    print_hyperparam_sweep(sweep_results)
    best_params = sweep_results[0]

    print("Training final OOD detector (One-Class SVM, standardized embeddings) "
          f"with layer={best_layer}, nu={best_params['nu']}, gamma={best_params['gamma']}, "
          f"kernel={best_params['kernel']}...")
    ood_detector = train_ood_detector(model, train_loader, device, layer=best_layer,
                                       standardize=True, nu=best_params["nu"],
                                       kernel=best_params["kernel"], gamma=best_params["gamma"])

    print("Fitting Mahalanobis class-conditional Gaussians on clean embeddings...")
    mahalanobis_params = train_mahalanobis_detector(model, train_loader, device)

    all_results = {}
    for shift_type in SHIFT_TYPES:
        print(f"Evaluating shift: {shift_type}...")
        all_results[shift_type] = run_evaluation(model, train_loader, ood_detector,
                                                  mahalanobis_params, temperature,
                                                  shift_type, device)

    print_results(all_results)