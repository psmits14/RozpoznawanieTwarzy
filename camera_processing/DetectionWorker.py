from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np


class DetectionWorker(QObject):
    finished = pyqtSignal(object)  # Sygnał emitowany po zakończeniu detekcji, przekazujący wyniki

    def __init__(self, detector, frame: np.ndarray):
        super().__init__()
        self.detector = detector    # Detektor do detekcji twarzy
        self.frame = frame          # Klatka obrazu (np. pojedyncza ramka z kamery lub wideo) do przetworzenia

    def run(self):
        """Metoda wykonywana w osobnym wątku do detekcji na podanej klatce"""
        try:
            # Wywołanie metody detekcji na klatce
            detections = self.detector.detect_faces(self.frame)
        except Exception as e:
            print(f"[ERROR] Detection error in thread: {e}")
            detections = []

        # Emitowanie sygnału z wynikami detekcji
        self.finished.emit(detections)
