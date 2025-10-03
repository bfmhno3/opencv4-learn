import cv2 as cv


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cats.jpg")
    blurred_img: cv.Mat = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
    canny_img: cv.Mat = cv.Canny(blurred_img, 125, 175)
    eroded_img: cv.Mat = cv.erode(canny_img, (3, 3), iterations=2)

    cv.imshow("Original", img)
    cv.imshow("Blurred", blurred_img)
    cv.imshow("Canny", canny_img)
    cv.imshow("Eroded", eroded_img)

    cv.waitKey(0)


if __name__ == "__main__":
    main()