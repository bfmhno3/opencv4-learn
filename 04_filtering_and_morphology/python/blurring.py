from pathlib import Path

import cv2 as cv


ROOT_DIR = Path(__file__).resolve().parents[2]
def main() -> None:
    img: cv.Mat = cv.imread(str(ROOT_DIR / "resources" / "photos" / "park.jpg"))
    average: cv.Mat = cv.blur(img, (3, 3))
    gaussian: cv.Mat = cv.GaussianBlur(img, (3, 3), 0)
    median: cv.Mat = cv.medianBlur(img, 3)
    bilateral: cv.Mat = cv.bilateralFilter(img, 3, 15, 15)

    cv.imshow("Original", img)
    cv.imshow("Average", average)
    cv.imshow("Gaussian", gaussian)
    cv.imshow("Median", median)
    cv.imshow("Bilateral", bilateral)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
