from PyQt6.QtCore import QTimer

from camera_processing.FaceApp import FaceApp
from camera_processing.VideoSource import VideoFileSource


class FaceAppController:
    def __init__(self, logger, face_app_ui, video_source):
        self.logger = logger
        self.face_app_ui = face_app_ui
        self.face_app = FaceApp(logger, face_app_ui, video_source)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update)

        # Obliczanie odpowiedniego interwału odświeżania
        if isinstance(video_source, VideoFileSource):
            target_fps = min(30, video_source._fps)  # ogranicz do max 30 FPS
            interval = max(10, int(1000 / target_fps))  # w ms
        else:
            interval = 33  # standardowe 30 FPS dla kamery

        self.timer.start(interval)  # użycie wyliczonego interwału

    def _update(self):
        self.face_app.update()

        # Aktualizacja kontrolek wideo (jeśli to plik wideo)
        if isinstance(self.face_app.video_source, VideoFileSource):
            current_frame = self.face_app.video_source.get_current_frame_position()
            total_frames = self.face_app.video_source.get_frame_count()
            self.face_app_ui.update_video_controls(current_frame, total_frames)

    def _toggle_play_pause(self):
        self.face_app.video_source.toggle_pause()
        self.face_app_ui.play_pause_btn.setText(
            "Odtwarzaj" if self.face_app.video_source.paused else "Pauza"
        )

    def _on_slider_moved(self, position):
        self.face_app.video_source.set_frame_position(position)
