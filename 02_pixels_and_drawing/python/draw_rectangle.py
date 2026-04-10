import cv2 as cv
import numpy as np


def main() -> None:
    blank = np.zeros((512, 512, 3), np.uint8)
    # cv.rectangle(blank, (0, 0), (256, 256), (0, 255, 0), thickness=2)
    cv.rectangle(blank, (0, 0), (256, 256), (255, 255, 255), thickness=cv.FILLED)

    cv.imshow("Blank", blank)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
