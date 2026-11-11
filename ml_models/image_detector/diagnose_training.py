
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from ml_models.image_detector.model import DeepfakeImageClassifier
from ml_models.image_detector.dataset import DeepfakeImageDataset


torch.manual_seed(42)
random.seed(42)


# Load the training dataset
dataset = DeepfakeImageDataset(split="train")


# Group images by their source video
videos = {
    0: {},
    1: {}
}

for index, (image_path, label) in enumerate(dataset.samples):

    video_name = image_path.parent.name
    label_value = label

    if video_name not in videos[label_value]:
        videos[label_value][video_name] = []

    videos[label_value][video_name].append(index)


# Select one image from each video folder
selected_indices = []

for label in [0, 1]:

    for video_name, indices in videos[label].items():

        selected_indices.append(random.choice(indices))


# Load the selected images
selected_samples = [
    dataset[index] for index in selected_indices
]

images = torch.stack([sample[0] for sample in selected_samples])
labels = torch.stack([sample[1] for sample in selected_samples])

print("Selected images:", images.shape)
print("REAL images:", (labels == 0).sum().item())
print("FAKE images:", (labels == 1).sum().item())


# Create a fresh model
model = DeepfakeImageClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=0.001
)


# Train repeatedly on the same 20 images
steps = 500

print("\nStarting 20-image overfitting diagnostic...\n")

for step in range(steps):

    model.train()

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    if (step + 1) % 50 == 0:

        model.eval()

        with torch.no_grad():

            predictions = torch.argmax(
                model(images),
                dim=1
            )

            accuracy = (
                (predictions == labels).float().mean() * 100
            )

        print(
            f"Step {step + 1}/{steps} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.2f}%"
        )


print("\nDiagnostic completed!")
