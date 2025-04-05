import cv2
import numpy as np

class CameraUI:
    def __init__(self, window_name="Aplikacja Kamery z Detekcją Twarzy"):
        self.window_name = window_name
        self.panel_width = 300
        self.max_faces_display = 5
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)

    def create_right_panel(self, width, height, detected_faces):
        """Tworzy panel boczny z wykrytymi twarzami"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (240, 240, 240)  # Szare tło

        # Nagłówek panelu
        cv2.putText(panel, "Wykryte twarze:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Wyświetlanie twarzy w panelu
        y_offset = 60
        max_face_width = width - 20
        max_face_height = (height - 100) // self.max_faces_display

        for i, face in enumerate(detected_faces[:self.max_faces_display]):
            try:
                if face.size == 0:
                    continue

                # Skalowanie zachowujące proporcje
                h, w = face.shape[:2]
                scale = min(max_face_width / w, max_face_height / h)
                face = cv2.resize(face, None, fx=scale, fy=scale)

                # Wyśrodkowanie
                x_pos = (width - face.shape[1]) // 2

                # Umieszczenie w panelu
                h, w = face.shape[:2]
                if y_offset + h < height:
                    panel[y_offset:y_offset + h, x_pos:x_pos + w] = face
                    cv2.putText(panel, f"Twarz {i + 1}", (x_pos, y_offset + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    y_offset += h + 40
            except Exception as e:
                continue

        return panel

    def display_frame(self, frame, detected_faces=[]):
        """Wyświetla klatkę z panelem bocznym"""
        right_panel = self.create_right_panel(self.panel_width, frame.shape[0], detected_faces)
        combined = np.hstack((frame, right_panel))
        cv2.imshow(self.window_name, combined)

    def should_close(self):
        """Sprawdza czy aplikacja powinna się zamknąć"""
        return (cv2.waitKey(1) == ord('q') or
                cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1)