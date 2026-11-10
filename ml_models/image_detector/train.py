
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from ml_models.image_detector.model import DeepfakeImageClassifier
from ml_models.image_detector.dataset import (
    DeepfakeImageDataset,
    create_data_loader
)


# Make training results more reproducible
torch.manual_seed(42)


# Training settings
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.001


def train_model():

    # Select CPU for training
    device = torch.device("cpu")

    # Load the datasets
    train_dataset = DeepfakeImageDataset(split="train")
    validation_dataset = DeepfakeImageDataset(split="validation")

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

    # Create the neural network
    model = DeepfakeImageClassifier().to(device)

    # Choose the loss function
    criterion = nn.CrossEntropyLoss()

    # Choose the optimizer
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Track the best validation accuracy
    best_validation_accuracy = -1.0

    # Where the best model will be saved
    model_path = Path(__file__).with_name("best_model.pth")

    print("\nStarting image model training...\n")

    # Repeat training for the chosen number of epochs
    for epoch in range(EPOCHS):

        # Training phase
        model.train()

        total_training_loss = 0.0
        correct_training_predictions = 0
        total_training_predictions = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Make predictions
            outputs = model(images)

            # Calculate prediction error
            loss = criterion(outputs, labels)

            # Calculate gradients
            loss.backward()

            # Update model weights
            optimizer.step()

            total_training_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct_training_predictions += (
                (predictions == labels).sum().item()
            )

            total_training_predictions += labels.size(0)

        average_training_loss = (
            total_training_loss / total_training_predictions
        )

        training_accuracy = (
            correct_training_predictions
            / total_training_predictions
        ) * 100

        # Validation phase
        model.eval()

        total_validation_loss = 0.0
        correct_validation_predictions = 0
        total_validation_predictions = 0

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

                correct_validation_predictions += (
                    (predictions == labels).sum().item()
                )

                total_validation_predictions += labels.size(0)

        average_validation_loss = (
            total_validation_loss / total_validation_predictions
        )

        validation_accuracy = (
            correct_validation_predictions
            / total_validation_predictions
        ) * 100

        # Display this epoch's results
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        print(f"Training Loss: {average_training_loss:.4f}")
        print(f"Training Accuracy: {training_accuracy:.2f}%")

        print(f"Validation Loss: {average_validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.2f}%")

        print("-" * 40)

        # Save the best model
        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "class_labels": {
                        "real": 0,
                        "fake": 1
                    }
                },
                model_path
            )

            print("Best model saved!")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_model()
