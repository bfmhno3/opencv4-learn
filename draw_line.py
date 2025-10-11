import cv2 as cv
import numpy as np


def main() -> None:
    blank: np.ndarray = np.zeros((512, 512, 3), dtype=np.uint8)
    cv.line(blank, (0, 0), (512, 512), (0, 255, 255), thickness=2)
    cv.imshow("blank", blank)

    cv.waitKey(0)


if __name__ == '__main__':
    main()