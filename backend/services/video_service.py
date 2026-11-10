
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, interval_seconds=1):

    # Convert the paths into Path objects
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Create the output folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the video
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get the video's frame rate
    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        capture.release()
        raise ValueError("Could not determine video frame rate.")

    # Calculate how many frames make up the chosen time interval
    frame_interval = max(1, int(round(fps * interval_seconds)))

    frame_number = 0
    saved_count = 0

    try:

        while True:

            # Read the next frame
            success, frame = capture.read()

            if not success:
                break

            # Save only frames matching our interval
            if frame_number % frame_interval == 0:

                output_path = output_dir / f"frame_{saved_count:04d}.jpg"

                saved = cv2.imwrite(str(output_path), frame)

                if not saved:
                    raise IOError(f"Could not save frame: {output_path}")

                saved_count += 1

            frame_number += 1

    finally:
        # Release the video resource
        capture.release()

    return saved_count


def extract_dataset_frames(dataset_root, frames_root, interval_seconds=1):

    dataset_root = Path(dataset_root)
    frames_root = Path(frames_root)

    video_groups = {
        "real": dataset_root / "original_sequences/youtube/c23/videos",
        "fake": dataset_root / "manipulated_sequences/Deepfakes/c23/videos"
    }

    total_videos = 0

    for label, video_folder in video_groups.items():

        if not video_folder.exists():
            raise FileNotFoundError(
                f"Video folder not found: {video_folder}"
            )

        video_paths = sorted(video_folder.glob("*.mp4"))

        print(f"\nProcessing {label} videos...")

        for video_path in tqdm(video_paths):

            video_name = video_path.stem

            output_folder = frames_root / label / video_name

            count = extract_frames(
                video_path,
                output_folder,
                interval_seconds
            )

            total_videos += 1

            print(f"{video_name}: {count} frames extracted")

    return total_videos
