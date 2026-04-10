from pathlib import Path
from typing import Tuple

import cv2 as cv


ROOT_DIR = Path(__file__).resolve().parents[2]


def rotate(img: cv.Mat, angle: float, rotation_point: Tuple[int, int] | None = None) -> cv.Mat:
    height, width = img.shape[:2]

    if rotation_point is None:
        rotation_point = (width // 2, height // 2)

    rotation_matrix = cv.getRotationMatrix2D(rotation_point, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotation_matrix, dimensions)


def main() -> None:
    img_path = ROOT_DIR / "resources" / "photos" / "cat.jpg"
    img: cv.Mat | None = cv.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")

    rotated_img: cv.Mat = rotate(img, 30)

    cv.imshow("Original", img)
    cv.imshow("Rotated", rotated_img)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
