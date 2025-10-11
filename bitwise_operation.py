import cv2 as cv
import numpy as np


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cats.jpg")
    blank: np.ndarray = np.zeros(img.shape, dtype=np.uint8)
    rectangle: cv.Mat = cv.rectangle(blank, (30, 30), (370, 370), (0, 255, 255), thickness=cv.FILLED)

    bitwise_and_img: cv.Mat = cv.bitwise_and(img, rectangle)
    bitwise_or_img: cv.Mat = cv.bitwise_or(img, rectangle)
    bitwise_xor_img: cv.Mat = cv.bitwise_xor(img, rectangle)
    bitwise_not_img: cv.Mat = cv.bitwise_not(img)

    cv.imshow("original_img", img)
    cv.imshow("rectangle", rectangle)
    cv.imshow("bitwise_and_img", bitwise_and_img)
    cv.imshow("bitwise_or_img", bitwise_or_img)
    cv.imshow("bitwise_xor_img", bitwise_xor_img)
    cv.imshow("bitwise_not_img", bitwise_not_img)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
