import cv2 as cv
import matplotlib.pyplot as plt


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cats.jpg")
    gray: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_hist: cv.Mat = cv.calcHist([img], [0], None, [256], [0, 256])

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins (Intensity)")
    plt.ylabel("# of Pixels")
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()


if __name__ == "__main__":
    main()
