import cv2 as cv
import numpy as np


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cat.jpg")
    gray_img: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred: cv.Mat = cv.GaussianBlur(gray_img, (3, 3), cv.BORDER_DEFAULT)
    canny_img: cv.Mat = cv.Canny(blurred, 125, 175)
    contours, hierarchies = cv.findContours(canny_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(img.shape[:2], dtype="uint8")
    cv.drawContours(blank, contours, -1, 255, 2)

    cv.imshow("Original", img)
    cv.imshow("Grayscale", gray_img)
    cv.imshow("Grayscale and Blurred", blurred)
    cv.imshow("Canny Edges", canny_img)
    cv.imshow("Contours", blank)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
