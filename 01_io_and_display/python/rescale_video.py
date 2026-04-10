import cv2 as cv


def main() -> None:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open default camera")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv.imshow("Rescaled", frame)
            if cv.waitKey(20) & 0xFF == ord("d"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
