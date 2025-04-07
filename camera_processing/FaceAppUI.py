from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QScrollArea, QSlider
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import cv2
import numpy as np
import os


class FaceAppUI(QWidget):
    def __init__(self, is_video_source=False):
        super().__init__()
        self.setWindowTitle("Aplikacja Kamery - Detekcja i Rozpoznawanie Twarzy")
        self.setMinimumSize(1000, 600)

        self.current_frame = None
        self.pending_face_crop = None
        self.on_add_face_callback = None
        self.on_prepare_face_crop_callback = None

        self.is_video_source = is_video_source

        self._setup_ui()

        if is_video_source:
            self._add_video_controls()

    def _setup_ui(self):
        # Obraz z kamery
        self.video_label = QLabel("Obraz z kamery")
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Scrollowana lista rozpoznań
        self.recognition_panel = QVBoxLayout()
        self.recognition_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll_widget = QWidget()
        scroll_widget.setLayout(self.recognition_panel)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        # Pole imienia i przycisk dodania
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Wpisz imię i naciśnij ENTER")
        self.name_input.returnPressed.connect(self._submit_face_name)
        self.name_input.hide()

        self.add_button = QPushButton("Dodaj nową twarz")
        self.add_button.clicked.connect(self._on_add_clicked)

        # Podgląd + akcje
        self.face_preview = QLabel()
        self.face_preview.setFixedSize(120, 120)
        self.face_preview.setVisible(False)

        self.confirm_button = QPushButton("Zatwierdź dodanie")
        self.confirm_button.clicked.connect(self._confirm_add_face)
        self.confirm_button.setVisible(False)

        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self._cancel_add_face)
        self.cancel_button.setVisible(False)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.face_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        button_row = QHBoxLayout()
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.confirm_button)
        preview_layout.addLayout(button_row)

        # Prawy panel
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Rozpoznane twarze:"))
        right_layout.addWidget(scroll_area)
        right_layout.addWidget(self.name_input)
        right_layout.addWidget(self.add_button)
        right_layout.addLayout(preview_layout)

        # Układ główny
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)
        self.setLayout(main_layout)

    def set_video_resolution(self, width: int, height: int):
        self.video_label.setFixedSize(width, height)

    def update_frame(self, frame: np.ndarray,
                     detected_faces: list[np.ndarray],
                     recognitions: list[dict]):
        self.current_frame = frame.copy()
        self._update_video_display(frame)
        self._update_face_comparisons(detected_faces, recognitions)

    def _update_video_display(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def _update_face_comparisons(self, faces, recognitions):
        while self.recognition_panel.count():
            item = self.recognition_panel.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, (cam_face, rec) in enumerate(zip(faces, recognitions)):
            label = f"{i + 1}. {rec.get('name', 'Nieznany')} - {rec.get('score', 0) * 100:.1f}%"
            ref_path = rec.get("reference")
            ref_face = cv2.imread(ref_path) if ref_path and os.path.exists(ref_path) else None
            widget = self._create_face_comparison_widget(cam_face, ref_face, label)
            self.recognition_panel.addWidget(widget)

    def _create_face_comparison_widget(self, cam_face: np.ndarray, ref_face: np.ndarray, label: str) -> QWidget:
        def to_pixmap(img, size=60):
            if img is None:
                blank = np.full((size, size, 3), 180, dtype=np.uint8)
                return QPixmap.fromImage(QImage(blank.data, size, size, size * 3, QImage.Format.Format_RGB888))

            h, w = img.shape[:2]
            scale = min(size / w, size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return QPixmap.fromImage(QImage(rgb.data, new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888))

        # Obrazki
        cam_pix = to_pixmap(cam_face)
        ref_pix = to_pixmap(ref_face)

        # Kamera
        cam_image = QLabel()
        cam_image.setPixmap(cam_pix)
        cam_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        cam_label = QLabel("Kamera")
        cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        cam_col = QVBoxLayout()
        cam_col.addWidget(cam_image)
        cam_col.addWidget(cam_label)

        # Wzorzec
        ref_image = QLabel()
        ref_image.setPixmap(ref_pix)
        ref_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ref_label = QLabel("Wzorzec")
        ref_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ref_col = QVBoxLayout()
        ref_col.addWidget(ref_image)
        ref_col.addWidget(ref_label)

        # Tekst
        text_label = QLabel(label)
        text_label.setStyleSheet("font-weight: bold; margin-left: 10px;")
        text_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Główne ułożenie
        layout = QHBoxLayout()
        layout.addLayout(cam_col)
        layout.addLayout(ref_col)
        layout.addWidget(text_label)

        container = QWidget()
        container.setLayout(layout)
        return container

    def _on_add_clicked(self):
        self.name_input.show()
        self.name_input.setFocus()

    def _submit_face_name(self):
        name = self.name_input.text().strip()
        if name and self.current_frame is not None:
            if self.on_prepare_face_crop_callback:
                crop = self.on_prepare_face_crop_callback(self.current_frame)
                if crop is not None:
                    self.pending_face_crop = (crop, name)
                    self._show_face_preview(crop)
        self.name_input.clear()
        self.name_input.hide()

    def _show_face_preview(self, crop: np.ndarray):
        max_size = 120
        h, w = crop.shape[:2]
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb.data, new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.face_preview.setPixmap(pixmap)
        self.face_preview.setVisible(True)
        self.confirm_button.setVisible(True)
        self.cancel_button.setVisible(True)

    def _confirm_add_face(self):
        if self.pending_face_crop and self.on_add_face_callback:
            crop, name = self.pending_face_crop
            self.on_add_face_callback(crop, name)
        self._reset_face_preview()

    def _cancel_add_face(self):
        self._reset_face_preview()

    def _reset_face_preview(self):
        self.face_preview.clear()
        self.face_preview.setVisible(False)
        self.confirm_button.setVisible(False)
        self.cancel_button.setVisible(False)
        self.pending_face_crop = None

    def _add_video_controls(self):
        # Dodajemy kontrolki do zarządzania wideo
        self.play_pause_btn = QPushButton("Pauza")
        self.play_pause_btn.clicked.connect(self._toggle_play_pause)

        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.sliderMoved.connect(self._on_slider_moved)

        video_controls = QHBoxLayout()
        video_controls.addWidget(self.play_pause_btn)
        video_controls.addWidget(self.video_slider)

        # Dodajemy kontrolki do istniejącego layoutu
        self.layout().insertLayout(1, video_controls)

    def _toggle_play_pause(self):
        # Ta metoda będzie wywoływana przez FaceApp
        pass

    def _on_slider_moved(self, position):
        # Ta metoda będzie wywoływana przez FaceApp
        pass

    def update_video_controls(self, current_frame, total_frames):
        if not self.is_video_source:
            return

        self.video_slider.setMaximum(total_frames)
        self.video_slider.setValue(current_frame)