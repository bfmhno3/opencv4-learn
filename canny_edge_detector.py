import cv2 as cv


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cats 2.jpg")
    blurred_img: cv.Mat = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
    gray_img: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_gray_img: cv.Mat = cv.cvtColor(blurred_img, cv.COLOR_BGR2GRAY)

    cv.imshow("Original", img)
    cv.imshow("Grayscale", gray_img)
    cv.imshow("Blurred", blurred_img)
    cv.imshow("Gray Blurred", blurred_gray_img)

    # Canny Edge Detection
    cv.imshow("Canny Edges (Original)", cv.Canny(img, 125, 175))
    cv.imshow("Canny Edges (Grayscale)", cv.Canny(gray_img, 125, 175))
    cv.imshow("Canny Edges (Blurred", cv.Canny(blurred_img, 125, 175))
    cv.imshow("Canny Edges (Gray Blurred)", cv.Canny(blurred_gray_img, 125, 175))

    cv.waitKey(0)


if __name__ == "__main__":
    main()