import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# CONFIDENCE BASED
# Runs a single forward pass, no extra overhead.

@torch.no_grad()
def evaluate_confidence(model, loader, device):
    """
    Returns per-sample entropy of the softmax distribution.
    High entropy = spread-out probabilities = uncertain prediction.
    """
    model.eval()
    all_entropy, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()

        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        all_entropy.extend(entropy)
        all_preds.extend(np.argmax(probs, axis=1))
        all_labels.extend(labels.numpy())

    return np.array(all_entropy), np.array(all_preds), np.array(all_labels)
    
# MONTE CARLO DROPOUT
# Runs N stochastic forward passes with dropout active, averages the softmax
# probabilities, then computes entropy over the mean distribution.

def evaluate_mc_dropout(model, loader, device, num_samples=15):
    """
    Returns per-sample predictive entropy averaged over MC samples.
    Requires model.forward() to accept mc_dropout=True.
    """
    model.eval()
    all_entropy, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            mc_probs = []

            for _ in range(num_samples):
                outputs = model(imgs, mc_dropout=True)
                mc_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

            # Shape: (num_samples, batch, classes) -> average over samples
            expected_probs = np.mean(np.stack(mc_probs, axis=0), axis=0)
            entropy = -np.sum(expected_probs * np.log(expected_probs + 1e-10), axis=1)

            all_entropy.extend(entropy)
            all_preds.extend(np.argmax(expected_probs, axis=1))
            all_labels.extend(labels.numpy())

    return np.array(all_entropy), np.array(all_preds), np.array(all_labels)

# TEMPERATURE-SCALED CONFIDENCE
# Learns a single scalar T on clean held-out logits (Guo et al., 2017), then
# uses the calibrated softmax entropy as the failure score.

def fit_temperature(model, val_loader, device, lr=0.01, max_iter=50):
    """
    Fits a scalar temperature that minimizes NLL of the model's logits on a
    clean held-out set. Returns the learned temperature (float > 0).
    """
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            logits_list.append(outputs.cpu())
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    nll_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def _closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return max(temperature.item(), 1e-3)


@torch.no_grad()
def evaluate_temperature_scaled(model, loader, device, temperature=1.0):
    """
    Same as evaluate_confidence, but divides logits by the learned temperature
    before computing softmax entropy — a calibrated confidence baseline.
    """
    model.eval()
    all_entropy, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs) / temperature
        probs = F.softmax(outputs, dim=1).cpu().numpy()

        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        all_entropy.extend(entropy)
        all_preds.extend(np.argmax(probs, axis=1))
        all_labels.extend(labels.numpy())

    return np.array(all_entropy), np.array(all_preds), np.array(all_labels)


# ENERGY SCORE
# Free-energy baseline (Liu et al., 2020): E(x) = -T * logsumexp(logits / T).
# Requires no extra fitting; higher energy = less in-distribution-like = likely fail.

@torch.no_grad()
def evaluate_energy(model, loader, device, temperature=1.0):
    model.eval()
    all_energy, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        energy = -temperature * torch.logsumexp(outputs / temperature, dim=1)

        all_energy.extend(energy.cpu().numpy())
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_energy), np.array(all_preds), np.array(all_labels)


# DISTANCE-BASED
# Extracts embeddings from the model's penultimate layer, fits a k-NN index on
# clean training data, then scores test samples by their distance to the k
# nearest training neighbors; Far from training data = likely OOD = likely fail.

@torch.no_grad()
def extract_embeddings_and_logits(model, loader, device, layer="final"):
    """Helper — extract embeddings AND classification predictions simultaneously.
    `layer` selects which stage of the network the embeddings come from
    ("layer2", "layer3", "layer4"/"final") so downstream detectors can compare
    representations of different depth."""
    model.eval()
    all_embeddings, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)

        outputs = model(imgs)
        embeddings = model.get_embeddings(imgs, layer=layer).cpu().numpy()

        all_embeddings.append(embeddings)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.array(all_preds), np.array(all_labels)


def fit_knn(train_embeddings, k=10):
    """Fit a k-NN index on clean training embeddings."""
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(train_embeddings)
    return knn


def evaluate_distance(model, train_loader, test_loader, device, k=10):
    """
    Returns mean k-NN distance to training set for each test sample.
    High distance = far from training distribution = likely to fail.
    """
    # Extract everything in a single, unified pass
    train_embeddings, _, _ = extract_embeddings_and_logits(model, train_loader, device)
    test_embeddings, test_preds, test_labels = extract_embeddings_and_logits(model, test_loader, device)

    # Compute k-NN distances
    knn = fit_knn(train_embeddings, k=k)
    distances, _ = knn.kneighbors(test_embeddings)
    mean_distances = distances.mean(axis=1)

    return mean_distances, test_preds, test_labels

# OOD DETECTOR
# Trains an unsupervised One-Class SVM on clean embeddings:
#   - Learns the boundary of the normal (in-distribution) data.
#   - Does not require shifted/OOD data during training.
# The inverted anomaly score indicates the likelihood of being OOD.

def train_ood_detector(model, clean_loader, device, layer="final", standardize=True,
                        nu=0.05, kernel="rbf", gamma="scale", max_train_samples=10000):
    """
    Trains an unsupervised One-Class SVM on CLEAN training data only.
    No shifted data is leaked during training.

    Returns a dict {"clf", "scaler", "layer"} — `scaler` is None when
    standardize=False, and is required by evaluate_ood to transform test
    embeddings the same way the training embeddings were transformed.
    """
    clean_embeddings, _, _ = extract_embeddings_and_logits(model, clean_loader, device, layer=layer)

    # Subsample if the training set is huge to speed up fitting
    if len(clean_embeddings) > max_train_samples:
        indices = np.random.choice(len(clean_embeddings), max_train_samples, replace=False)
        clean_embeddings = clean_embeddings[indices]

    scaler = None
    if standardize:
        scaler = StandardScaler()
        clean_embeddings = scaler.fit_transform(clean_embeddings)

    # gamma='scale' or 'auto' handles high-dimensional space
    clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    clf.fit(clean_embeddings)
    return {"clf": clf, "scaler": scaler, "layer": layer}


def evaluate_ood(model, ood_detector, test_loader, device):
    """
    Scores each test sample based on its distance from the clean distribution.
    Invert score because lower decision function values mean MORE out-of-distribution.
    `ood_detector` is the dict returned by train_ood_detector.
    """
    clf = ood_detector["clf"]
    scaler = ood_detector["scaler"]
    layer = ood_detector["layer"]

    # Extract everything in a single, unified pass, from the same layer used at training time
    test_embeddings, test_preds, test_labels = extract_embeddings_and_logits(model, test_loader, device, layer=layer)

    if scaler is not None:
        test_embeddings = scaler.transform(test_embeddings)

    # score_samples returns the log density; inverting it means higher score = higher OOD likelihood.
    ood_scores = -clf.score_samples(test_embeddings)
    return ood_scores, test_preds, test_labels


def compare_embedding_layers(model, train_loader, val_shifted_loader, device,
                              layers=("layer2", "layer3", "final"), standardize=True,
                              nu=0.05, kernel="rbf", gamma="scale"):
    """
    Trains an OCSVM detector on embeddings from each candidate layer and reports
    AUROC (against actual model failures) on a held-out shifted validation set.
    Returns a list of {"layer", "auroc"} dicts, best layer first.
    """
    results = []
    for layer in layers:
        detector = train_ood_detector(model, train_loader, device, layer=layer,
                                       standardize=standardize, nu=nu, kernel=kernel, gamma=gamma)
        scores, preds, targets = evaluate_ood(model, detector, val_shifted_loader, device)
        failures = (preds != targets).astype(int)
        auroc = roc_auc_score(failures, scores) if 0 < failures.sum() < len(failures) else float("nan")
        results.append({"layer": layer, "auroc": auroc})

    results.sort(key=lambda r: (r["auroc"] if r["auroc"] == r["auroc"] else -1), reverse=True)
    return results


def sweep_ocsvm_hyperparameters(model, train_loader, val_shifted_loader, device,
                                 layer="final", standardize=True,
                                 nus=(0.01, 0.05, 0.1), gammas=("scale", "auto"),
                                 kernels=("rbf",)):
    """
    Grid search over OCSVM hyperparameters (nu, gamma, kernel), scoring each
    combination by AUROC against actual model failures on a held-out shifted
    validation set. Returns a list of result dicts sorted best-first.
    """
    results = []
    for kernel in kernels:
        for nu in nus:
            for gamma in gammas:
                detector = train_ood_detector(model, train_loader, device, layer=layer,
                                               standardize=standardize, nu=nu,
                                               kernel=kernel, gamma=gamma)
                scores, preds, targets = evaluate_ood(model, detector, val_shifted_loader, device)
                failures = (preds != targets).astype(int)
                auroc = roc_auc_score(failures, scores) if 0 < failures.sum() < len(failures) else float("nan")
                results.append({"kernel": kernel, "nu": nu, "gamma": gamma, "auroc": auroc})

    results.sort(key=lambda r: (r["auroc"] if r["auroc"] == r["auroc"] else -1), reverse=True)
    return results


# MAHALANOBIS DISTANCE
# Fits class-conditional Gaussians on clean training embeddings:
# one mean per class, one shared ("tied") covariance across all classes.
# Score = minimum Mahalanobis distance to any class mean; far from every
# known cluster = likely OOD / likely to fail.

def train_mahalanobis_detector(model, train_loader, device, layer="final", eps=1e-6):
    """
    Returns a dict {"means", "precision", "layer"}:
      - means: {class_id: mean_embedding}
      - precision: inverse of the shared, regularized covariance matrix
    """
    embeddings, _, labels = extract_embeddings_and_logits(model, train_loader, device, layer=layer)
    classes = np.unique(labels)

    means = {}
    centered = []
    for c in classes:
        class_embeddings = embeddings[labels == c]
        mean_c = class_embeddings.mean(axis=0)
        means[c] = mean_c
        centered.append(class_embeddings - mean_c)

    centered = np.concatenate(centered, axis=0)
    cov = np.cov(centered, rowvar=False)
    cov += eps * np.eye(cov.shape[0])  # regularize for numerical stability
    precision = np.linalg.inv(cov)

    return {"means": means, "precision": precision, "layer": layer}


def evaluate_mahalanobis(model, mahalanobis_params, test_loader, device):
    """
    Scores each sample by its minimum Mahalanobis distance to any class mean,
    using the shared precision matrix fit on clean training data.
    """
    means = mahalanobis_params["means"]
    precision = mahalanobis_params["precision"]
    layer = mahalanobis_params["layer"]

    embeddings, test_preds, test_labels = extract_embeddings_and_logits(model, test_loader, device, layer=layer)

    class_ids = list(means.keys())
    mean_matrix = np.stack([means[c] for c in class_ids])  # (num_classes, dim)

    distances = np.stack([
        np.einsum("ij,jk,ik->i", embeddings - mean_matrix[i], precision, embeddings - mean_matrix[i])
        for i in range(len(class_ids))
    ], axis=1)  # (num_samples, num_classes)

    min_distances = distances.min(axis=1)
    return min_distances, test_preds, test_labels