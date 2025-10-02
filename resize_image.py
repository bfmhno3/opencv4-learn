import cv2 as cv


def resize_image(image: cv.Mat, scale: int) -> cv.Mat:
    height, width = image.shape[:2]
    dimensions = (int(width * scale), int(height * scale))
    interpolation = cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR
    return cv.resize(image, dimensions, interpolation=interpolation)


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cat.jpg")
    cv.imshow("Original", img)
    resized_img: cv.Mat = resize_image(img, 1.5)
    cv.imshow("Resized", resized_img)
    cv.waitKey(0)


if __name__ == "__main__":
    main()

