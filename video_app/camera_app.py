import cv2


class CameraApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        self.window_name = "Camera App"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting...")
                break

            cv2.imshow(self.window_name, frame)

            # Check for both key press AND window close
            key = cv2.waitKey(1)
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = CameraApp()
    app.run()