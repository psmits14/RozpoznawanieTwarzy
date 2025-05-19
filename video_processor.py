import logging.config
import sys
import warnings
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from camera_processing.FaceAppUI import FaceAppUI
from camera_processing.VideoSource import VideoFileSource
from camera_processing.FaceAppController import FaceAppController

from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)
VIDEO_FOLDER = Path(r"C:\Users\julia\OneDrive\Obrazy\Camera Roll")


def configure_logging():
    logging.config.fileConfig("config/logging.conf")
    return logging.getLogger('api')


class VideoProcessor:
    def __init__(self):
        self.logger = configure_logging()
        self.app = QApplication(sys.argv)

        self.video_files = self._get_video_files()
        self.current_video_index = 0
        self.face_app_controller = None

        # Ustawienia timera
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._save_and_move_next)
        self.video_duration = 1 * 60 * 1000  # minuty w milisekundach

        self._process_next_video()

    def _get_video_files(self):
        """Zwraca listę plików wideo do przetworzenia"""
        if not VIDEO_FOLDER.exists():
            self.logger.error(f"Folder nie istnieje: {VIDEO_FOLDER}")
            return []

        extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv"]
        video_files = []
        for ext in extensions:
            video_files.extend(sorted(VIDEO_FOLDER.glob(ext)))

        self.logger.info(f"Znaleziono {len(video_files)} plików wideo")
        return video_files

    def _process_next_video(self):
        """Przetwarza kolejny plik wideo"""
        if self.current_video_index >= len(self.video_files):
            self.logger.info("Koniec przetwarzania plików")
            sys.exit(0)

        video_file = self.video_files[self.current_video_index]
        self.logger.info(
            f"Przetwarzanie pliku {self.current_video_index}/{len(self.video_files)}: {video_file.name}")

        try:
            self._play_video(str(video_file))
            self.video_timer.start(self.video_duration)  # Uruchom timer
        except Exception as e:
            self.logger.error(f"Błąd: {str(e)}")
            self.current_video_index += 1
            QTimer.singleShot(0, self._process_next_video)

    def _play_video(self, video_path):
        """Odtwarza pojedynczy plik wideo"""
        if self.face_app_controller:
            self.face_app_controller.stop()
            if hasattr(self.face_app_controller, 'face_app_ui'):
                self.face_app_controller.face_app_ui.close()

        video_source = VideoFileSource(video_path)
        face_app_ui = FaceAppUI(is_video_source=True)

        self.face_app_controller = FaceAppController(
            self.logger,
            face_app_ui,
            video_source,
            recognition_threshold=0.6,  # Domyślna wartość
            sound_enabled=False
        )

        face_app_ui.show()

    def run(self):
        sys.exit(self.app.exec())

    def _save_and_move_next(self):
        """Zapisuje log i przechodzi do następnego pliku"""
        if self.face_app_controller and hasattr(self.face_app_controller, 'face_app'):
            self.face_app_controller.face_app.save_recognition_log()
        self.current_video_index += 1
        QTimer.singleShot(1000, self._process_next_video)  # Małe opóźnienie przed następnym plikiem

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()