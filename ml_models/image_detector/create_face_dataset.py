from pathlib import Path
import cv2


# Define the input and output folders
SOURCE_ROOT = Path("dataset/splits")
OUTPUT_ROOT = Path("dataset/face_splits")

# Load the pretrained YuNet face detector
MODEL_PATH = (
    "ml_models/face_detection/"
    "face_detection_yunet_2023mar.onnx"
)

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


def crop_largest_face(image):

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # Select the largest detected face
    face = max(
        faces,
        key=lambda item: item[2] * item[3]
    )

    x, y, face_width, face_height = face[:4]

    # Add some surrounding area around the face
    center_x = x + face_width / 2
    center_y = y + face_height / 2

    crop_size = max(face_width, face_height) * 1.5

    left = max(0, int(center_x - crop_size / 2))
    top = max(0, int(center_y - crop_size / 2))

    right = min(width, int(center_x + crop_size / 2))
    bottom = min(height, int(center_y + crop_size / 2))

    return image[top:bottom, left:right]


def create_face_dataset():

    totals = {}

    for split in ["train", "validation", "test"]:

        for class_name in ["real", "fake"]:

            source_class = SOURCE_ROOT / split / class_name
            output_class = OUTPUT_ROOT / split / class_name

            if not source_class.exists():
                raise FileNotFoundError(
                    f"Dataset folder not found: {source_class}"
                )

            totals[(split, class_name)] = {
                "processed": 0,
                "saved": 0,
                "skipped": 0
            }

            # Preserve the original video-folder structure
            for video_folder in sorted(source_class.iterdir()):

                if not video_folder.is_dir():
                    continue

                output_folder = output_class / video_folder.name
                output_folder.mkdir(parents=True, exist_ok=True)

                for image_path in sorted(video_folder.glob("*.jpg")):

                    image = cv2.imread(str(image_path))

                    if image is None:
                        totals[(split, class_name)]["skipped"] += 1
                        continue

                    face_crop = crop_largest_face(image)

                    totals[(split, class_name)]["processed"] += 1

                    if face_crop is None or face_crop.size == 0:

                        totals[(split, class_name)]["skipped"] += 1
                        continue

                    output_path = output_folder / image_path.name

                    saved = cv2.imwrite(
                        str(output_path),
                        face_crop
                    )

                    if saved:
                        totals[(split, class_name)]["saved"] += 1
                    else:
                        totals[(split, class_name)]["skipped"] += 1

    print("\nFACE DATASET CREATION RESULTS")
    print("=" * 50)

    for (split, class_name), counts in totals.items():

        print(f"\n{split.upper()} - {class_name.upper()}")
        print("Processed:", counts["processed"])
        print("Saved:", counts["saved"])
        print("Skipped:", counts["skipped"])

    print("\nFace dataset creation completed!")


if __name__ == "__main__":
    create_face_dataset()
