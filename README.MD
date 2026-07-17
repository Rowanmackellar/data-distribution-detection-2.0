## Data Distribution && Model Failure Detection

This repository implements and evaluates seven different failure detection paradigms to predict when a ResNet-18 model will fail under distribution shift on the CIFAR-10 dataset. Six shift types are covered: **blur**, **noise**, **brightness/contrast**, **rotation**, **occlusion**, and **JPEG compression**.

## SETUP:

1: create a virtual environment;

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2: train the model;

    python train.py

3: evaluate the detectors;

    python evaluate.py

4: inspect failures;

    python inspect_failures.py

## DETECTION METHODS IMPLEMENTED:

- **Confidence-Based:** Softmax distribution entropy (single forward pass).
- **Temperature-Scaled Confidence:** Softmax entropy after calibrating logits with a learned scalar temperature (Guo et al., 2017), fit on clean held-out data.
- **Energy Score:** Free-energy baseline (Liu et al., 2020), `E(x) = -T · logsumexp(logits / T)`, requiring no extra fitting.
- **Distance-Based:** Mean k-NN distance from clean training embeddings.
- **Monte Carlo Dropout:** Predictive entropy averaged over 15 stochastic forward passes.
- **OOD Anomaly Detection:** Unsupervised One-Class SVM trained on clean, standardized embeddings.
- **Mahalanobis Distance:** Class-conditional Gaussians fit on clean embeddings (Lee et al., 2018); scored by minimum distance to any class mean.

Each method scores how likely a prediction is to be wrong. `evaluate.py` turns that score into an AUROC against actual model errors, across every shift type.

## OOD DETECTOR TUNING

The One-Class SVM detector is the most sensitive to setup, so `evaluate.py` runs two tuning passes before the final evaluation (both validated on the blur shift, held out from the reported results):

- **Layer comparison:** embeddings are compared across `layer2`, `layer3`, and the final pooled layer, since intermediate features may separate shifted data better than the final representation.
- **Hyperparameter sweep:** a grid search over `nu`, `gamma`, and `kernel` on the best-performing layer.

The winning layer/hyperparameter combination is used to train the OOD detector reported in the results table below.

## Results

AUROC for detecting actual model errors, per method, per shift (0.5 = random, 1.0 = perfect). Run `python evaluate.py` to regenerate this table:

| Shift               | Confidence | Temp-Scaled | Energy | Distance | MC Dropout | OOD (SVM) | Mahalanobis |
|---------------------|:----------:|:-----------:|:------:|:--------:|:----------:|:---------:|:-----------:|
| Blur                |            |             |        |          |            |           |             |
| Noise               |            |             |        |          |            |           |             |
| Brightness/Contrast |            |             |        |          |            |           |             |
| Rotation            |            |             |        |          |            |           |             |
| Occlusion           |            |             |        |          |            |           |             |
| JPEG                |            |             |        |          |            |           |             |

## Additional Failure Inspection

In the 4th step we highlight the core focus of this repository-
Here are examples of images where our ResNet-18 model was highly confident, but completely wrong due to distribution shifts:

![Confident Failures Gallery](plots/confident_failures.png)