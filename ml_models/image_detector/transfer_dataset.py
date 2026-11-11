import torch
from torchvision import transforms

from ml_models.image_detector.dataset import DeepfakeImageDataset


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNetImageDataset(DeepfakeImageDataset):

    def __init__(
        self,
        dataset_dir="dataset/face_splits",
        split="train",
        augment=False
    ):

        super().__init__(
            dataset_dir=dataset_dir,
            split=split,
            augment=augment
        )

        self.normalization = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    def __getitem__(self, index):

        image, label = super().__getitem__(index)

        image = self.normalization(image)

        return image, label
