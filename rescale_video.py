import cv2 as cv


def change_resolution(cap: cv.VideoCapture, width: int, height: int) -> None:
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


def main() -> None:
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow('Rescaled', frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(0)


if __name__ == '__main__':
    main()
