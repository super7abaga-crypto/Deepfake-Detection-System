from PIL import Image, ImageOps
import numpy as np
import torch

def validate_image(image_path):
    """
    Check whether an uploaded file is a valid image.
    """

    try:
        with Image.open(image_path) as image:

            image.verify()

        return {
            "valid": True,
            "message": "Image is valid."
        }

    except Exception:
        return {
            "valid": False,
            "message": "Invalid or corrupted image."
        }

from PIL import Image, ImageOps


def preprocess_image(image_path):
    """
    Prepare an image for machine-learning analysis.
    """

    with Image.open(image_path) as image:

        # Convert the image to RGB
        rgb_image = image.convert("RGB")

        # Resize while preserving the original proportions
        resized_image = ImageOps.contain(
            rgb_image,
            (224, 224),
            method=Image.Resampling.LANCZOS
        )

        # Create a square background
        processed_image = Image.new(
            "RGB",
            (224, 224),
            (127, 127, 127)
        )

        # Centre the resized image
        x = (224 - resized_image.width) // 2
        y = (224 - resized_image.height) // 2

        processed_image.paste(resized_image, (x, y))

        return processed_image


def image_to_tensor(image):
    """
    Convert a processed image into a PyTorch tensor.
    """

    # Convert image pixels into numerical values
    image_array = np.array(image, dtype=np.float32)

    # Scale pixel values from 0-255 to 0-1
    image_array = image_array / 255.0

    # Convert the NumPy array into a PyTorch tensor
    image_tensor = torch.from_numpy(image_array)

    # Change dimensions from HWC to CHW
    image_tensor = image_tensor.permute(2, 0, 1)

    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
