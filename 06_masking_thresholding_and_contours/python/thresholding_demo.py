from pathlib import Path

import cv2 as cv


ROOT_DIR = Path(__file__).resolve().parents[2]
def main() -> None:
    img: cv.Mat = cv.imread(str(ROOT_DIR / "resources" / "photos" / "cat.jpg"))
    gray_img: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, binary_low = cv.threshold(gray_img, 85, 255, cv.THRESH_BINARY)
    _, binary_mid = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
    _, binary_high = cv.threshold(gray_img, 170, 255, cv.THRESH_BINARY)
    _, binary_inverse_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY_INV)

    cv.imshow("Original", img)
    cv.imshow("Grayscale", gray_img)
    cv.imshow("Binary (85)", binary_low)
    cv.imshow("Binary (127)", binary_mid)
    cv.imshow("Binary (170)", binary_high)
    cv.imshow("Binary Inverse (127)", binary_inverse_img)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
