import cv2 as cv


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/park.jpg")
    average: cv.Mat = cv.blur(img, (3, 3))
    gaussian: cv.Mat = cv.GaussianBlur(img, (3, 3), 0)
    median: cv.Mat = cv.medianBlur(img, 3)
    bilateral: cv.Mat = cv.bilateralFilter(img, 3, 15, 15)

    cv.imshow("Original", img)
    cv.imshow("Average", average)
    cv.imshow("Gaussian", gaussian)
    cv.imshow("Median", median)
    cv.imshow("Bilateral", bilateral)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
