from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_resnet18(num_classes: int, pretrained: bool, cifar_stem: bool = False) -> nn.Module:
    weights: Optional[ResNet18_Weights] = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def maybe_freeze_backbone(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("fc"):
            parameter.requires_grad = False


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    cifar_stem: bool = False,
) -> nn.Module:
    normalized_name = model_name.lower()
    if normalized_name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    if normalized_name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, cifar_stem=cifar_stem)
    raise ValueError(f"Unsupported model_name: {model_name}")
