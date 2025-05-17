from PyQt6.QtCore import QTimer
from camera_processing.FaceApp import FaceApp
from camera_processing.VideoSource import VideoFileSource

class FaceAppController:
    def __init__(self, logger, face_app_ui, video_source):
        self.logger = logger
        self.face_app_ui = face_app_ui
        self.face_app = FaceApp(logger, face_app_ui, video_source)

        # Połączenia dla wideo
        if isinstance(video_source, VideoFileSource):
            self.face_app_ui.play_pause_btn.clicked.connect(self._toggle_play_pause)
            self.face_app_ui.video_slider.sliderReleased.connect(self._on_slider_released)
            self.face_app_ui.video_slider.sliderPressed.connect(self._on_slider_pressed)

            self.face_app_ui.set_video_fps(video_source._fps)
            self.face_app_ui.video_slider.setMaximum(video_source.get_frame_count())

        # QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)

        if isinstance(video_source, VideoFileSource):
            target_fps = min(30, video_source._fps)
            interval = max(10, int(1000 / target_fps))  # 10–33ms
        else:
            interval = 33  # Kamera – standardowo 30 FPS

        self.timer.start(interval)

    def _update(self):
        if not self.face_app.video_source.paused:
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

    def _on_slider_pressed(self):
        self.face_app_ui._slider_being_dragged = True

    def _on_slider_released(self):
        self.face_app_ui._slider_being_dragged = False
        frame_num = self.face_app_ui.video_slider.value()
        self.face_app.video_source.set_frame_position(frame_num)
