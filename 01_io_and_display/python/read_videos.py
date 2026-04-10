from pathlib import Path

import cv2 as cv
import os
from typing import List


ROOT_DIR = Path(__file__).resolve().parents[2]


def get_video_paths(dir_path: str) -> List[str]:
    return [
        video_path
        for video_path in (
            os.path.join(dir_path, video_name) for video_name in os.listdir(dir_path)
        )
        if os.path.isfile(video_path)
    ]


def play_video(video_path: str) -> None:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    window_name = os.path.basename(video_path)
    has_shown_window = False
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv.imshow(window_name, frame)
            has_shown_window = True
            if cv.waitKey(20) & 0xFF == ord("d"):
                break
    finally:
        cap.release()
        if has_shown_window:
            cv.destroyWindow(window_name)


def main() -> None:
    video_paths: List[str] = get_video_paths(str(ROOT_DIR / "resources" / "videos"))

    for video_path in video_paths:
        play_video(video_path)


if __name__ == "__main__":
    main()
