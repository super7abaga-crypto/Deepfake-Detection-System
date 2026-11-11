
import torch

from ml_models.image_detector.model import DeepfakeImageClassifier
from ml_models.image_detector.dataset import (
    DeepfakeImageDataset,
    create_data_loader
)


# Load the latest trained model
model = DeepfakeImageClassifier()

checkpoint = torch.load(
    "ml_models/image_detector/latest_model.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])


# Examine training and validation predictions
for split in ["train", "validation"]:

    dataset = DeepfakeImageDataset(
        split=split,
        augment=False
    )

    loader = create_data_loader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    results = {
        "real": [],
        "fake": []
    }

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            for probability, label in zip(probabilities, labels):

                class_name = "real" if label.item() == 0 else "fake"

                results[class_name].append(probability)

    print(f"\n{split.upper()} CONFIDENCE RESULTS")
    print("-" * 40)

    for class_name in ["real", "fake"]:

        class_probabilities = torch.stack(results[class_name])

        average_real = class_probabilities[:, 0].mean().item()
        average_fake = class_probabilities[:, 1].mean().item()

        print(f"\nActual {class_name.upper()} images:")
        print(f"Average REAL probability: {average_real:.4f}")
        print(f"Average FAKE probability: {average_fake:.4f}")
