import cv2 as cv
import os
from typing import List


def get_image_path(dir_path: str) -> List[str]:
    return [
        image_path
        for image_path in (
            os.path.join(dir_path, name) for name in os.listdir(dir_path)
        )
        if os.path.isfile(image_path)
    ]


def main() -> None:
    image_paths: List[str] = get_image_path("resources/photos")

    for image_path in image_paths:
        img: cv.Mat = cv.imread(image_path)
        cv.imshow(f"{os.path.basename(image_path)}", img)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
