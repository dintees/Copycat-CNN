import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetModel(nn.Module):
    """ResNet-based model."""

    def __init__(self, num_classes=10):
        super(ResNetModel, self).__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)