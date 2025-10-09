import cv2 as cv
import numpy as np


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/park.jpg")
    blank_channel = np.zeros(img.shape[:2], dtype="uint8")
    blue_channel, green_channel, red_channel = cv.split(img)

    blue_only = cv.merge([blue_channel, blank_channel, blank_channel])
    green_only = cv.merge([blank_channel, green_channel, blank_channel])
    red_only = cv.merge([blank_channel, blank_channel, red_channel])

    cv.imshow("Original", img)
    cv.imshow("Blue Channel", blue_channel)
    cv.imshow("Green Channel", green_channel)
    cv.imshow("Red Channel", red_channel)
    cv.imshow("Blue Only", blue_only)
    cv.imshow("Green Only", green_only)
    cv.imshow("Red Only", red_only)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
