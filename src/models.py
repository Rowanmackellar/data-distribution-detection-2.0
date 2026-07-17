import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
 
 
class ResNetCIFAR(nn.Module):
    """ResNet-18 optimized for 32x32 CIFAR images with MC Dropout support."""
 
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        self.model = models.resnet18(weights=None)
 
        # Modify first layer for low-resolution inputs, to fit 32x32
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
 
        self.dropout = nn.Dropout(p=dropout_rate)
        self.model.fc = nn.Linear(512, num_classes)
 
    def _encode(self, x):
        m = self.model
        x = m.relu(m.bn1(m.conv1(x)))
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        return torch.flatten(x, 1)
 
    def forward(self, x, mc_dropout=False):
        x = self._encode(x)
        # mc_dropout=True forces dropout active regardless of model.eval(),
        # enabling stochastic forward passes for uncertainty estimation.
        if mc_dropout:
            x = F.dropout(x, p=self.dropout.p, training=True)
        else:
            x = self.dropout(x)
        return self.model.fc(x)
 
    def get_embeddings(self, x, layer="final"):
        """Return pooled feature vectors from a given point in the network.

        layer:
            "layer2" -> 128-d features after the second residual stage
            "layer3" -> 256-d features after the third residual stage
            "layer4" final layer -> 512-d features before the classifier head (default)

        Intermediate layers are exposed so OOD detectors, the One Class SVM for example;
        can be compared across representations of different depth/abstraction.
        """
        with torch.no_grad():
            m = self.model
            x = m.relu(m.bn1(m.conv1(x)))
            x = m.layer1(x)
            x = m.layer2(x)
            if layer == "layer2":
                return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
            x = m.layer3(x)
            if layer == "layer3":
                return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
            x = m.layer4(x)
            if layer in ("layer4", "final"):
                return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
            raise ValueError(f"Unknown layer: {layer}")