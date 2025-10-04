import cv2 as cv


def rotate(img: cv.Mat, angle: float, rotation_point: tuple[int] = None) -> cv.Mat:
    height, width = img.shape[:2]

    if rotation_point is None:
        rotation_point = (width // 2, height // 2)

    rotation_matrix = cv.getRotationMatrix2D(rotation_point, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotation_matrix, dimensions)


def main() -> None:
    img: cv.Mat = cv.imread("resources/photos/cat.jpg")
    rotated_img: cv.Mat = rotate(img, 30)

    cv.imshow("Original", img)
    cv.imshow("Rotated", rotated_img)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
