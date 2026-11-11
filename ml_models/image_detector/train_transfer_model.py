from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from ml_models.image_detector.transfer_model import DeepfakeResNet18
from ml_models.image_detector.dataset import create_data_loader
from ml_models.image_detector.transfer_dataset import ResNetImageDataset


torch.manual_seed(42)


# Training settings
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.001


def train_transfer_model():

    device = torch.device("cpu")

    # Load face-cropped datasets
    train_dataset = ResNetImageDataset(
        dataset_dir="dataset/face_splits",
        split="train",
        augment=False
    )

    validation_dataset = ResNetImageDataset(
        dataset_dir="dataset/face_splits",
        split="validation"
    )

    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = create_data_loader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Load pretrained ResNet18
    model = DeepfakeResNet18().to(device)

    criterion = nn.CrossEntropyLoss()

    # Train only the new classification layer
    optimizer = Adam(
        model.model.fc.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float("inf")

    model_path = Path(__file__).with_name(
        "resnet18_normalized_best_model.pth"
    )

    latest_model_path = Path(__file__).with_name(
        "resnet18_normalized_latest_model.pth"
    )

    print("\nStarting ResNet18 transfer learning...\n")

    for epoch in range(EPOCHS):

        # Training phase
        model.train()

        total_training_loss = 0.0
        correct_training = 0
        total_training = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_training_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct_training += (
                (predictions == labels).sum().item()
            )

            total_training += labels.size(0)

        average_training_loss = (
            total_training_loss / total_training
        )

        training_accuracy = (
            correct_training / total_training
        ) * 100

        # Validation phase
        model.eval()

        total_validation_loss = 0.0
        correct_validation = 0
        total_validation = 0

        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                total_validation_loss += (
                    loss.item() * images.size(0)
                )

                predictions = torch.argmax(outputs, dim=1)

                correct_validation += (
                    (predictions == labels).sum().item()
                )

                total_validation += labels.size(0)

        average_validation_loss = (
            total_validation_loss / total_validation
        )

        validation_accuracy = (
            correct_validation / total_validation
        ) * 100

        # Display results
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Training Loss: {average_training_loss:.4f}")
        print(f"Training Accuracy: {training_accuracy:.2f}%")
        print(f"Validation Loss: {average_validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.2f}%")
        print("-" * 40)

        # Save checkpoint information
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "validation_accuracy": validation_accuracy,
            "validation_loss": average_validation_loss,
            "class_labels": {
                "real": 0,
                "fake": 1
            },
            "architecture": "ResNet18",
            "dataset": "face_crops"
        }

        torch.save(checkpoint, latest_model_path)

        # Save best validation checkpoint
        if average_validation_loss < best_validation_loss:

            best_validation_loss = average_validation_loss

            torch.save(checkpoint, model_path)

            print("Best ResNet18 model saved!")

    print("\nTransfer learning completed!")

    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best model: {model_path}")
    print(f"Latest model: {latest_model_path}")


if __name__ == "__main__":
    train_transfer_model()
