
import torch
import torch.nn as nn
from torch.optim import Adam

from ml_models.image_detector.model import DeepfakeImageClassifier
from ml_models.image_detector.dataset import DeepfakeImageDataset


torch.manual_seed(42)

# 1. Load the complete training dataset
dataset = DeepfakeImageDataset(split="train")

images = []
labels = []

for image, label in dataset:
    images.append(image)
    labels.append(label)

# Combine individual images into one large batch
images = torch.stack(images)
labels = torch.stack(labels)

print("Dataset loaded successfully!")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# 2. Create the model
model = DeepfakeImageClassifier()

# 3. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 4. Train repeatedly on the same complete dataset
steps = 500

print("\nStarting full-dataset diagnostic...\n")

for step in range(steps):

    model.train()

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    predictions = torch.argmax(outputs, dim=1)

    accuracy = (predictions == labels).float().mean() * 100

    if (step + 1) % 10 == 0 or step == 0:
        print(
            f"Step {step + 1}/{steps} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.2f}%"
        )

print("\nDiagnostic completed!")
