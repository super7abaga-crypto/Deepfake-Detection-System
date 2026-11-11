
from pathlib import Path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from backend.services.image_service import (
    preprocess_image,
    image_to_tensor
)


class DeepfakeImageDataset(Dataset):

    def __init__(
        self,
        dataset_dir="dataset/splits",
        split="train",
        augment=False
    ):

        self.dataset_dir = Path(dataset_dir) / split

        # Augmentation is allowed only for training data
        self.augment = augment and split == "train"

        # Assign numerical labels to our classes
        self.class_labels = {
            "real": 0,
            "fake": 1
        }

        self.samples = []

        # Find images in both class folders
        for class_name, label in self.class_labels.items():

            class_folder = self.dataset_dir / class_name

            if not class_folder.exists():
                raise FileNotFoundError(
                    f"Dataset folder not found: {class_folder}"
                )

            for image_path in sorted(class_folder.rglob("*")):

                if image_path.suffix.lower() in [
                    ".jpg", ".jpeg", ".png", ".webp"
                ]:

                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(
                f"No images found in dataset split: {self.dataset_dir}"
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        # Prepare the image
        image = preprocess_image(image_path)

        # Randomly mirror training images
        if self.augment and random.random() < 0.5:

            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        # Convert the image into a tensor
        tensor = image_to_tensor(image)

        # Remove the extra batch dimension
        tensor = tensor.squeeze(0)

        return tensor, torch.tensor(label, dtype=torch.long)


def create_data_loader(dataset, batch_size=4, shuffle=True):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
