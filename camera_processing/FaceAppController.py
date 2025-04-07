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
        self.timer.start(30)

        if isinstance(video_source, VideoFileSource):
            self.face_app_ui.play_pause_btn.clicked.connect(self._toggle_play_pause)
            self.face_app_ui.video_slider.sliderMoved.connect(self._on_slider_moved)

    def _update(self):
        self.face_app.update()
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