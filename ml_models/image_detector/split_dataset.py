
from pathlib import Path
import random
import shutil


def split_dataset(frames_root, output_root):

    frames_root = Path(frames_root)
    output_root = Path(output_root)

    # Define the five source-video pairs
    video_pairs = [
        ("183", "253"),
        ("469", "481"),
        ("585", "599"),
        ("672", "720"),
        ("866", "878")
    ]

    # Shuffle pairs in a reproducible way
    random.seed(42)
    random.shuffle(video_pairs)

    # Assign pairs to training, validation and testing
    splits = {
        "train": video_pairs[:3],
        "validation": video_pairs[3:4],
        "test": video_pairs[4:]
    }

    # Copy the images into their assigned splits
    for split_name, pairs in splits.items():

        for original_a, original_b in pairs:

            # Copy both original videos
            real_videos = [original_a, original_b]

            for video_id in real_videos:

                source_folder = frames_root / "real" / video_id

                destination_folder = (
                    output_root / split_name / "real" / video_id
                )

                destination_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                for image_path in source_folder.glob("*.jpg"):

                    shutil.copy2(
                        image_path,
                        destination_folder / image_path.name
                    )

            # Copy both manipulated videos
            fake_videos = [
                f"{original_a}_{original_b}",
                f"{original_b}_{original_a}"
            ]

            for video_id in fake_videos:

                source_folder = frames_root / "fake" / video_id

                destination_folder = (
                    output_root / split_name / "fake" / video_id
                )

                destination_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                for image_path in source_folder.glob("*.jpg"):

                    shutil.copy2(
                        image_path,
                        destination_folder / image_path.name
                    )

    return splits


if __name__ == "__main__":

    root = Path.home() / "Datasets/FaceForensicsPlusPlus/frames"

    output = Path("dataset/splits")

    result = split_dataset(root, output)

    print("\nDataset splitting completed!")

    for split_name, pairs in result.items():
        print(f"{split_name}: {len(pairs)} source pairs")
