import cv2 as cv


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/park.jpg")
    gray_img: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hsv_img: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab_img: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    rgb_img: cv.Mat = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    cv.imshow("Original", img)
    cv.imshow("BGR to Grayscale", gray_img)
    cv.imshow("BGR to HSV", hsv_img)
    cv.imshow("BGR to LAB", lab_img)
    cv.imshow("BGR to RGB", rgb_img)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
