import cv2 as cv
import os
from typing import List


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
    window_name = os.path.basename(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow(window_name, frame)
        if cv.waitKey(20) & 0xFF == ord("d"):
            break
    cap.release()
    cv.destroyWindow(window_name)


def main() -> None:
    video_paths: List[str] = get_video_paths("resources/videos")

    for video_path in video_paths:
        play_video(video_path)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
