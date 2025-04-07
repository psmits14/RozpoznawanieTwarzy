import time

import cv2
from abc import ABC, abstractmethod


class VideoSource(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def release(self):
        pass

    @abstractmethod
    def get_frame_size(self):
        pass


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
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height


class VideoFileSource(VideoSource):
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć pliku wideo: {file_path}")
        self._processing_width = 1280
        self.paused = False
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = max(1, self.cap.get(cv2.CAP_PROP_FPS))  # Minimum 1 FPS
        self._last_valid_frame = None  # Do przechowywania ostatniej klatki

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # Skalowanie do optymalnej rozdzielczości
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
        self.paused = not self.paused

    def get_current_frame_position(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def get_frame_count(self):
        return self._frame_count

    def set_frame_position(self, frame_num):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)