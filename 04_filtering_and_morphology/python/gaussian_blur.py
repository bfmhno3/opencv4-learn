from pathlib import Path

import cv2 as cv


ROOT_DIR = Path(__file__).resolve().parents[2]
def main() -> None:
    img: cv.Mat = cv.imread(str(ROOT_DIR / "resources" / "photos" / "cats.jpg"))
    blurred_img: cv.Mat = cv.GaussianBlur(img, (11, 11), cv.BORDER_DEFAULT)
    cv.imshow("Original", img)
    cv.imshow("Blurred", blurred_img)
    cv.waitKey(0)


if __name__ == "__main__":
    main()