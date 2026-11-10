from PIL import Image


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