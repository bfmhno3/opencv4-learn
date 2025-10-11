import cv2 as cv
import numpy as np


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cats.jpg")
    mask: np.ndarray = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, thickness=cv.FILLED)

    masked_img: cv.Mat = cv.bitwise_and(img, img, mask=mask)

    cv.imshow("original", img)
    cv.imshow("mask", mask)
    cv.imshow("masked_img", masked_img)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
