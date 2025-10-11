import cv2 as cv
import numpy as np


def main() -> None:
    blank: np.ndarray = np.zeros((512, 512, 3), dtype=np.uint8)
    cv.putText(blank, "Hello World", (75, 250), cv.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 2)
    cv.imshow("blank", blank)

    cv.waitKey(0)


if __name__ == "__main__":
    main()