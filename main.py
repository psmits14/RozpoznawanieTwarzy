# main.py

import logging.config
import sys
import warnings
from PyQt6.QtWidgets import QApplication, QMessageBox

from camera_processing.FaceAppUI import FaceAppUI
from camera_processing.StartUI import StartUI
from camera_processing.VideoSource import CameraSource, VideoFileSource
from camera_processing.FaceAppController import FaceAppController

from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def configure_logging():
    """Konfiguracja loggera na podstawie pliku konfiguracyjnego"""
    logging.config.fileConfig("config/logging.conf")
    return logging.getLogger('api')

class Application:
    def __init__(self):
        self.logger = configure_logging()
        self.app = QApplication(sys.argv)

        self.start_ui = StartUI()
        self.start_ui.on_camera_selected = self._start_camera
        self.start_ui.on_video_selected = self._start_video

        self.face_app_controller = None
        self.app.aboutToQuit.connect(self._on_exit)
        self.start_ui.show()

    def _start_camera(self):
        """Uruchamia aplikację z kamerą jako źródłem obrazu"""
        try:
            video_source = CameraSource()
            self._start_face_app(video_source, is_video_source=False)
        except Exception as e:
            QMessageBox.critical(self.start_ui, "Błąd", str(e))

    def _start_video(self, video_path):
        """Uruchamia aplikację z plikiem wideo jako źródłem obrazu"""
        try:
            video_source = VideoFileSource(video_path)
            self._start_face_app(video_source, is_video_source=True)
        except Exception as e:
            QMessageBox.critical(self.start_ui, "Błąd", str(e))

    def _start_face_app(self, video_source, is_video_source):
        """Uruchamia główną aplikację rozpoznawania twarzy z wybranym źródłem obrazu"""
        self.start_ui.hide()
        threshold = self.start_ui.get_threshold_value()
        sound_on = self.start_ui.is_sound_enabled()
        print("[DEBUG] sound_enabled =", sound_on)
        face_app_ui = FaceAppUI(is_video_source=is_video_source)
        face_app_ui.on_back_callback = self._handle_back_to_start

        self.face_app_controller = FaceAppController(
            self.logger,
            face_app_ui,
            video_source,
            recognition_threshold=threshold,
            sound_enabled=sound_on
        )

        face_app_ui.show()

    def _handle_back_to_start(self):
        """Powrót z aplikacji rozpoznawania do ekranu początkowego"""
        if self.face_app_controller:
            self.face_app_controller.stop()

            # Zamknij okno interfejsu rozpoznawania
            self.face_app_controller.face_app_ui.close()

            self.face_app_controller = None

        self.start_ui.show()

    def _on_exit(self):
        """Obsługa zamykania aplikacji"""
        if self.face_app_controller:
            self.face_app_controller.stop()

    def run(self):
        """Uruchomienie pętli aplikacji Qt"""
        sys.exit(self.app.exec())


if __name__ == "__main__":
    app = Application()
    app.run()