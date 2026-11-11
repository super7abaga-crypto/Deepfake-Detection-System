import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DeepfakeResNet18(nn.Module):

    def __init__(self):

        super().__init__()

        # Load ResNet18 with pretrained ImageNet weights
        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Freeze the pretrained feature-extraction layers
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Replace the original 1000-class output layer
        input_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            input_features,
            2
        )

    def forward(self, images):

        return self.model(images)
