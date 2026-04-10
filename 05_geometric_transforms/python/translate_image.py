from pathlib import Path

import cv2 as cv
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
def translate(img: cv.Mat, x: int, y: int) -> cv.Mat:
    translation_mat = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(img, translation_mat, (img.shape[1], img.shape[0]))


def main() -> None:
    img: cv.Mat = cv.imread(str(ROOT_DIR / "resources" / "photos" / "cat.jpg"))
    translated_img: cv.Mat = translate(img, 100, 100)

    cv.imshow("Original", img)
    cv.imshow("Translated", translated_img)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
