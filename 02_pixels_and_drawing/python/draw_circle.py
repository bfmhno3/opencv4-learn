import cv2 as cv
import numpy as np


def main() -> None:
    blank = np.zeros((512, 512, 3), dtype="uint8")

    cv.circle(blank, (256, 256), 256, (255, 255, 255), thickness=cv.FILLED)
    cv.imshow("Blank", blank)

    cv.waitKey(0)


if __name__ == "__main__":
    main()
