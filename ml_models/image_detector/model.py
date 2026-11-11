import torch
import torch.nn as nn


class DeepfakeImageClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Reduce the feature map to one value per channel
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification layer
        self.classifier = nn.Linear(64, 2)

    def forward(self, image):

        x = self.features(image)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        output = self.classifier(x)

        return output
