from PyQt6.QtCore import QTimer
from camera_processing.FaceApp import FaceApp
from camera_processing.VideoSource import VideoFileSource, CameraSource

class FaceAppController:
    def __init__(self, logger, face_app_ui, video_source, recognition_threshold=0.5):
        self.logger = logger
        self.face_app_ui = face_app_ui
        self.video_source = video_source
        self.face_app = FaceApp(
            logger,
            face_app_ui,
            video_source,
            recognition_threshold=recognition_threshold
        )

        # Slider drag state
        self.face_app_ui._slider_being_dragged = False

        # --- Połączenia dla VideoFileSource ---
        if isinstance(video_source, VideoFileSource):
            self.face_app_ui.play_pause_btn.clicked.connect(self._toggle_play_pause)
            self.face_app_ui.video_slider.sliderPressed.connect(self._on_slider_pressed)
            self.face_app_ui.video_slider.sliderReleased.connect(self._on_slider_released)

            self.face_app_ui.set_video_fps(video_source._fps)
            self.face_app_ui.video_slider.setMaximum(video_source.get_frame_count())

        # --- QTimer ---
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)

        if isinstance(video_source, VideoFileSource):
            fps = min(30, video_source._fps)
            self.interval = max(10, int(1000 / fps))
        else:
            self.interval = 33  # Dla kamery

        self.timer.start(self.interval)

    def _update(self):
        # Kamera: zawsze aktualizuj
        # Wideo: tylko jeśli nie zapauzowane
        if isinstance(self.video_source, CameraSource):
            self.face_app.update()
        elif isinstance(self.video_source, VideoFileSource):
            if not self.video_source.paused:
                self.face_app.update()

            current_frame = self.video_source.get_current_frame_position()
            total_frames = self.video_source.get_frame_count()

            if not self.face_app_ui._slider_being_dragged:
                self.face_app_ui.update_video_controls(current_frame, total_frames)

    def _toggle_play_pause(self):
        if isinstance(self.video_source, VideoFileSource):
            self.video_source.toggle_pause()
            self.face_app_ui.play_pause_btn.setText(
                "Odtwarzaj" if self.video_source.paused else "Pauza"
            )

    def _on_slider_pressed(self):
        self.face_app_ui._slider_being_dragged = True

    def _on_slider_released(self):
        self.face_app_ui._slider_being_dragged = False
        frame_num = self.face_app_ui.video_slider.value()
        self.video_source.set_frame_position(frame_num)

    def stop(self):
        if self.face_app:
            self.face_app.save_recognition_log()