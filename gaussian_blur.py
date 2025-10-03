import cv2 as cv


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cats.jpg")
    blurred_img: cv.Mat = cv.GaussianBlur(img, (11, 11), cv.BORDER_DEFAULT)
    cv.imshow("Original", img)
    cv.imshow("Blurred", blurred_img)
    cv.waitKey(0)


if __name__ == "__main__":
    main()