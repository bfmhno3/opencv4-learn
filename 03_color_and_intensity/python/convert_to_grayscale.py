from pathlib import Path

import cv2 as cv


ROOT_DIR = Path(__file__).resolve().parents[2]
def main() -> None:
    img: cv.Mat = cv.imread(str(ROOT_DIR / "resources" / "photos" / "cat.jpg"))
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Original", img)
    cv.imshow("Grayscale", gray_img)
    cv.waitKey(0)


if __name__ == "__main__":
    main()