import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# CONFIDENCE BASED
@torch.no_grad()
def evaluate_confidence(model, loader, device):
    """Returns per-sample entropy of the softmax distribution."""
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
def evaluate_mc_dropout(model, loader, device, num_samples=15):
    """Returns per-sample predictive entropy averaged over MC samples."""
    model.eval()
    all_entropy, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            mc_probs = []

            for _ in range(num_samples):
                outputs = model(imgs, mc_dropout=True)
                mc_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

            expected_probs = np.mean(np.stack(mc_probs, axis=0), axis=0)
            entropy = -np.sum(expected_probs * np.log(expected_probs + 1e-10), axis=1)

            all_entropy.extend(entropy)
            all_preds.extend(np.argmax(expected_probs, axis=1))
            all_labels.extend(labels.numpy())

    return np.array(all_entropy), np.array(all_preds), np.array(all_labels)

# TEMPERATURE-SCALED CONFIDENCE
def fit_temperature(model, val_loader, device, lr=0.01, max_iter=50):
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

# DISTANCE-BASED & FEATURE EXTRACTION
@torch.no_grad()
def extract_embeddings_and_logits(model, loader, device, layer="final", l2_normalize=False):
    """Extract embeddings and classification predictions, with optional L2 feature normalization."""
    model.eval()
    all_embeddings, all_preds, all_labels = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)

        outputs = model(imgs)
        emb = model.get_embeddings(imgs, layer=layer)
        if l2_normalize:
            emb = F.normalize(emb, p=2, dim=1)
        embeddings = emb.cpu().numpy()

        all_embeddings.append(embeddings)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.array(all_preds), np.array(all_labels)

def fit_knn(train_embeddings, k=10):
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(train_embeddings)
    return knn

def evaluate_distance(model, train_loader, test_loader, device, k=10, l2_normalize=False):
    train_embeddings, _, _ = extract_embeddings_and_logits(model, train_loader, device, l2_normalize=l2_normalize)
    test_embeddings, test_preds, test_labels = extract_embeddings_and_logits(model, test_loader, device, l2_normalize=l2_normalize)

    knn = fit_knn(train_embeddings, k=k)
    distances, _ = knn.kneighbors(test_embeddings)
    mean_distances = distances.mean(axis=1)

    return mean_distances, test_preds, test_labels

# OOD DETECTOR (One-Class SVM)
def train_ood_detector(model, clean_loader, device, layer="final", standardize=True,
                        l2_normalize=False, nu=0.05, kernel="rbf", gamma="scale", max_train_samples=10000):
    clean_embeddings, _, _ = extract_embeddings_and_logits(model, clean_loader, device, layer=layer, l2_normalize=l2_normalize)

    if len(clean_embeddings) > max_train_samples:
        indices = np.random.choice(len(clean_embeddings), max_train_samples, replace=False)
        clean_embeddings = clean_embeddings[indices]

    scaler = None
    if standardize:
        scaler = StandardScaler()
        clean_embeddings = scaler.fit_transform(clean_embeddings)

    clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    clf.fit(clean_embeddings)
    return {"clf": clf, "scaler": scaler, "layer": layer, "l2_normalize": l2_normalize}

def evaluate_ood(model, ood_detector, test_loader, device):
    clf = ood_detector["clf"]
    scaler = ood_detector["scaler"]
    layer = ood_detector["layer"]
    l2_normalize = ood_detector.get("l2_normalize", False)

    test_embeddings, test_preds, test_labels = extract_embeddings_and_logits(
        model, test_loader, device, layer=layer, l2_normalize=l2_normalize
    )

    if scaler is not None:
        test_embeddings = scaler.transform(test_embeddings)

    ood_scores = -clf.score_samples(test_embeddings)
    return ood_scores, test_preds, test_labels

def compare_embedding_layers(model, train_loader, val_shifted_loader, device,
                              layers=("layer2", "layer3", "final"), standardize=True, l2_normalize=False,
                              nu=0.05, kernel="rbf", gamma="scale"):
    results = []
    for layer in layers:
        detector = train_ood_detector(model, train_loader, device, layer=layer,
                                       standardize=standardize, l2_normalize=l2_normalize,
                                       nu=nu, kernel=kernel, gamma=gamma)
        scores, preds, targets = evaluate_ood(model, detector, val_shifted_loader, device)
        failures = (preds != targets).astype(int)
        auroc = roc_auc_score(failures, scores) if 0 < failures.sum() < len(failures) else float("nan")
        results.append({"layer": layer, "auroc": auroc})

    results.sort(key=lambda r: (r["auroc"] if r["auroc"] == r["auroc"] else -1), reverse=True)
    return results

def sweep_ocsvm_hyperparameters(model, train_loader, val_shifted_loader, device,
                                 layer="final", standardize=True, l2_normalize=False,
                                 nus=(0.01, 0.05, 0.1), gammas=("scale", "auto"),
                                 kernels=("rbf",)):
    results = []
    for kernel in kernels:
        for nu in nus:
            for gamma in gammas:
                detector = train_ood_detector(model, train_loader, device, layer=layer,
                                               standardize=standardize, l2_normalize=l2_normalize,
                                               nu=nu, kernel=kernel, gamma=gamma)
                scores, preds, targets = evaluate_ood(model, detector, val_shifted_loader, device)
                failures = (preds != targets).astype(int)
                auroc = roc_auc_score(failures, scores) if 0 < failures.sum() < len(failures) else float("nan")
                results.append({"kernel": kernel, "nu": nu, "gamma": gamma, "auroc": auroc})

    results.sort(key=lambda r: (r["auroc"] if r["auroc"] == r["auroc"] else -1), reverse=True)
    return results

# MAHALANOBIS DISTANCE
def train_mahalanobis_detector(model, train_loader, device, layer="final", l2_normalize=False, eps=1e-6):
    embeddings, _, labels = extract_embeddings_and_logits(model, train_loader, device, layer=layer, l2_normalize=l2_normalize)
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
    cov += eps * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov)

    return {"means": means, "precision": precision, "layer": layer, "l2_normalize": l2_normalize}

def evaluate_mahalanobis(model, mahalanobis_params, test_loader, device):
    means = mahalanobis_params["means"]
    precision = mahalanobis_params["precision"]
    layer = mahalanobis_params["layer"]
    l2_normalize = mahalanobis_params.get("l2_normalize", False)

    embeddings, test_preds, test_labels = extract_embeddings_and_logits(
        model, test_loader, device, layer=layer, l2_normalize=l2_normalize
    )

    class_ids = list(means.keys())
    mean_matrix = np.stack([means[c] for c in class_ids])

    distances = np.stack([
        np.einsum("ij,jk,ik->i", embeddings - mean_matrix[i], precision, embeddings - mean_matrix[i])
        for i in range(len(class_ids))
    ], axis=1)

    min_distances = distances.min(axis=1)
    return min_distances, test_preds, test_labels
