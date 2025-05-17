from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np


class DetectionWorker(QObject):
    finished = pyqtSignal(object)  # Signal z wynikami detekcji (lista)

    def __init__(self, detector, frame: np.ndarray):
        super().__init__()
        self.detector = detector
        self.frame = frame

    def run(self):
        try:
            detections = self.detector.detect_faces(self.frame)
        except Exception as e:
            print(f"[ERROR] Detection error in thread: {e}")
            detections = []

        self.finished.emit(detections)
