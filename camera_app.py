import cv2
import numpy as np
import sys
import logging.config
import yaml
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler


class FaceCameraApp:
    def __init__(self):
        # Inicjalizacja kamery
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Błąd: Nie można uruchomić kamery.")
            return

        # Konfiguracja detektora twarzy
        sys.path.append('video_app')
        logging.config.fileConfig("config/logging.conf")
        self.logger = logging.getLogger('api')

        with open('config/model_conf.yaml') as f:
            model_conf = yaml.load(f, Loader=yaml.FullLoader)

        # Ładowanie modelu detekcji twarzy
        model_path = 'models'
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name = model_conf[scene][model_category]

        self.logger.info('Ładowanie modelu detekcji twarzy...')
        try:
            faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
            self.model, self.cfg = faceDetModelLoader.load_model()
            self.faceDetModelHandler = FaceDetModelHandler(self.model, 'cpu', self.cfg)
            self.logger.info('Model detekcji twarzy załadowany pomyślnie!')
        except Exception as e:
            self.logger.error('Błąd ładowania modelu!')
            self.logger.error(e)
            sys.exit(-1)

        # Konfiguracja okna
        self.window_name = "Aplikacja Kamery z Zaawansowaną Detekcją Twarzy"
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
        self.panel_width = 300  # Szerokość panelu z twarzami

        self.detected_faces = []
        self.max_faces_display = 5

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Błąd: Nie można odczytać obrazu.")
                break

            # Lustrzane odbicie
            frame = cv2.flip(frame, 1)

            # Wykrywanie twarzy
            try:
                dets = self.faceDetModelHandler.inference_on_image(frame)

                # Czyszczenie poprzednio wykrytych twarzy
                self.detected_faces = []

                # Zaznaczanie twarzy na obrazie i zapisywanie nowych twarzy
                for det in dets[:self.max_faces_display]:  # Ogranicz do max_faces_display
                    # Pobierz współrzędne bounding box (pierwsze 4 wartości)
                    box = list(map(int, det[:4]))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                    # Wycinanie twarzy z obrazu (z zabezpieczeniem przed wyjściem poza zakres)
                    x1, y1, x2, y2 = box
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    if x2 > x1 and y2 > y1:  # Sprawdzenie czy wymiary są poprawne
                        face_img = frame[y1:y2, x1:x2]
                        self.detected_faces.append(face_img)

            except Exception as e:
                self.logger.error('Błąd detekcji twarzy!')
                self.logger.error(str(e))
                continue

            # Tworzenie panelu z twarzami
            right_panel = self.create_right_panel(self.panel_width, frame.shape[0])

            # Łączenie obrazów
            combined = np.hstack((frame, right_panel))

            cv2.imshow(self.window_name, combined)

            if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def create_right_panel(self, width, height):
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (240, 240, 240)  # Szare tło

        # Nagłówek panelu
        cv2.putText(panel, "Wykryte twarze:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Wyświetlanie twarzy w panelu
        y_offset = 60
        max_face_width = width - 20
        max_face_height = (height - 100) // self.max_faces_display

        for i, face in enumerate(self.detected_faces):
            try:
                # Sprawdzenie czy obraz twarzy jest poprawny
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
                self.logger.warning(f"Błąd przetwarzania twarzy: {str(e)}")
                continue

        return panel


if __name__ == "__main__":
    app = FaceCameraApp()
    app.run()