import cv2
from abc import ABC, abstractmethod

# Abstrakcyjna klasa bazowa dla źródeł wideo
class VideoSource(ABC):
    @abstractmethod
    def read(self):
        """Odczytuje jedną klatkę ze źródła wideo."""
        pass

    @abstractmethod
    def release(self):
        """Zwalnia zasoby związane ze źródłem wideo."""
        pass

    @abstractmethod
    def get_frame_size(self):
        """Zwraca rozmiar klatek (szerokość, wysokość)."""
        pass

# Źródło wideo z kamery
class CameraSource(VideoSource):
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Nie można uruchomić kamery")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def get_frame_size(self):
        """Zwraca rozdzielczość kamery."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height


# Źródło wideo z pliku
class VideoFileSource(VideoSource):
    def __init__(self, file_path):
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć pliku wideo: {file_path}")
        self._processing_width = 1280
        self.paused = False
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = max(1, self.cap.get(cv2.CAP_PROP_FPS))  # Minimum 1 FPS

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # Skalowanie
        if frame.shape[1] > self._processing_width:
            scale = self._processing_width / frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        return True, frame

    def release(self):
        self.cap.release()

    def get_frame_size(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def toggle_pause(self):
        """Przełącza pauzę/wznowienie odtwarzania"""
        self.paused = not self.paused

    def get_current_frame_position(self):
        """Zwraca numer aktualnie odtwarzanej klatki"""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def get_frame_count(self):
        """Zwraca całkowitą liczbę klatek w pliku"""
        return self._frame_count

    def set_frame_position(self, frame_num):
        """Ustawia numer klatki do odtworzenia"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
