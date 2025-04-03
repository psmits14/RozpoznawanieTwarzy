import cv2
import numpy as np


class FaceCameraApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Błąd: Nie można uruchomić kamery.")
            return

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.window_name = "Aplikacja Kamery z Detekcją Twarzy"
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)  # Normalne okno z paskiem tytułowym
        cv2.resizeWindow(self.window_name, 1200, 600)  # Wymiary startowe

        self.detected_faces = []
        self.max_faces_display = 5

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Błąd: Nie można odczytać obrazu.")
                break

            height, width = frame.shape[:2]
            offset = int(width * 0.25)  # Przesunięcie obrazu o 25%
            camera_view = frame[:, :width - offset]

            # Wykrywanie twarzy
            gray = cv2.cvtColor(camera_view, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            # Zaznaczanie twarzy
            for (x, y, w, h) in faces:
                cv2.rectangle(camera_view, (x, y), (x + w, y + h), (255, 0, 0), 2)

            self.update_faces_list(faces, camera_view)
            right_panel = self.create_right_panel(width - offset, height)

            # Łączenie obrazów z zachowaniem proporcji
            if camera_view.shape[0] != right_panel.shape[0]:
                right_panel = cv2.resize(right_panel, (right_panel.shape[1], camera_view.shape[0]))

            combined = np.hstack((camera_view, right_panel))

            cv2.imshow(self.window_name, combined)

            # Obsługa zamknięcia okna
            if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_faces_list(self, faces, frame):
        self.detected_faces = []
        for i, (x, y, w, h) in enumerate(faces[:self.max_faces_display]):
            face_img = frame[y:y + h, x:x + w]
            self.detected_faces.append(face_img)

    def create_right_panel(self, width, height):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (240, 240, 240)  # Jasnoszare tło

        # Nagłówek panelu
        cv2.putText(panel, "Wykryte twarze:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Wyświetlanie twarzy w panelu
        y_offset = 60
        face_width = width - 20  # Szerokość dla twarzy

        for i, face in enumerate(self.detected_faces):
            try:
                # Skalowanie zachowujące proporcje
                h, w = face.shape[:2]
                scale = min((face_width) / w, (height // self.max_faces_display) / h)
                face = cv2.resize(face, None, fx=scale, fy=scale)

                # Wyśrodkowanie w poziomie
                x_pos = (width - face.shape[1]) // 2

                # Umieszczenie twarzy w panelu
                h, w = face.shape[:2]
                if y_offset + h < height:  # Sprawdzenie czy zmieści się w panelu
                    panel[y_offset:y_offset + h, x_pos:x_pos + w] = face

                    # Podpis pod twarzą
                    cv2.putText(panel, f"Twarz {i + 1}", (x_pos, y_offset + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    y_offset += h + 40
            except Exception as e:
                print(f"Błąd przetwarzania twarzy: {e}")
                continue

        return panel


if __name__ == "__main__":
    app = FaceCameraApp()
    app.run()