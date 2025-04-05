import cv2
from .FaceDetector import FaceDetector
from .CameraUI import CameraUI


class FaceCameraApp:
    def __init__(self, logger):
        self.logger = logger
        self.camera = self._initialize_camera()
        self.detector = FaceDetector(logger)
        self.ui = CameraUI()
        self.detected_faces = []

    def _initialize_camera(self):
        """Inicjalizacja kamery"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Nie można uruchomić kamery.")
            raise RuntimeError("Nie można uruchomić kamery")
        return cap

    def process_frame(self, frame):
        """Przetwarza pojedynczą klatkę"""
        # Lustrzane odbicie
        frame = cv2.flip(frame, 1)

        # Wykrywanie twarzy
        detections = self.detector.detect_faces(frame)
        self.detected_faces = []

        # Zaznaczanie i wycinanie twarzy
        for det in detections:
            box = list(map(int, det[:4]))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            # Wycinanie twarzy z zabezpieczeniem przed wyjściem poza zakres
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 > x1 and y2 > y1:
                face_img = frame[y1:y2, x1:x2]
                self.detected_faces.append(face_img)

        return frame

    def run(self):
        """Główna pętla aplikacji"""
        while True:
            ret, frame = self.camera.read()
            if not ret:
                self.logger.error("Nie można odczytać obrazu z kamery.")
                break

            processed_frame = self.process_frame(frame)
            self.ui.display_frame(processed_frame, self.detected_faces)

            if self.ui.should_close():
                break

        self.camera.release()
        cv2.destroyAllWindows()