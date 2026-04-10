from pathlib import Path

import cv2 as cv


ROOT_DIR = Path(__file__).resolve().parents[2]
def main() -> None:
    img: cv.Mat = cv.imread(str(ROOT_DIR / "resources" / "photos" / "cat.jpg"))
    flipped_horizontally_img: cv.Mat = cv.flip(img, 1)
    flipped_vertically_img: cv.Mat = cv.flip(img, 0)
    flipped_both_img: cv.Mat = cv.flip(img, -1)

    cv.imshow("Original", img)
    cv.imshow("Flipped Horizontally", flipped_horizontally_img)
    cv.imshow("Flipped Vertically", flipped_vertically_img)
    cv.imshow("Flipped Both Image", flipped_both_img)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
