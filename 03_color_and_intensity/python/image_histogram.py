from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    img_path = ROOT_DIR / "resources" / "photos" / "cats.jpg"
    img: cv.Mat | None = cv.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")

    gray: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_hist: cv.Mat = cv.calcHist([gray], [0], None, [256], [0, 256])

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins (Intensity)")
    plt.ylabel("# of Pixels")
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()


if __name__ == "__main__":
    main()
