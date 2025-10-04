import cv2 as cv


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cat.jpg")
    cropped_img: cv.Mat = img[50:200, 200:400]

    cv.imshow("Original", img)
    cv.imshow("Cropped", cropped_img)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
