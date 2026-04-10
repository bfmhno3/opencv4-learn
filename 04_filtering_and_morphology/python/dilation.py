from pathlib import Path

import cv2 as cv


ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    img_path = ROOT_DIR / "resources" / "photos" / "cats.jpg"
    img: cv.Mat | None = cv.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    blurred_img: cv.Mat = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
    canny_img: cv.Mat = cv.Canny(blurred_img, 125, 175)
    dilated_img: cv.Mat = cv.dilate(canny_img, kernel, iterations=3)

    cv.imshow("Original", img)
    cv.imshow("Blurred", blurred_img)
    cv.imshow("Canny", canny_img)
    cv.imshow("Dilated", dilated_img)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
