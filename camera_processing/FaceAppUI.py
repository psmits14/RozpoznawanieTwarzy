from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QScrollArea, QSlider, QSizePolicy, QApplication
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

        self._target_width = 1280  # Docelowa szerokość wideo
        self._target_height = 720  # Docelowa wysokość wideo
        self._setup_window_geometry()

        if is_video_source:
            self._add_video_controls()

    def _setup_ui(self):
        # Główny układ pionowy dla lewej strony (obraz + kontrolki wideo)
        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)

        # Obraz z kamery
        self.video_label = QLabel("Obraz z kamery")
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_column.addWidget(self.video_label, stretch=1)

        # Kontrolki wideo (będą dodawane tylko dla źródła wideo)
        self.video_controls = QHBoxLayout()
        self.video_controls.setContentsMargins(5, 0, 5, 5)
        left_column.addLayout(self.video_controls)

        # Scrollowana lista rozpoznań (prawy panel - pozostaje bez zmian)
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

        # Prawy panel (pozostaje bez zmian)
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Rozpoznane twarze:"))
        right_layout.addWidget(scroll_area)
        right_layout.addWidget(self.name_input)
        right_layout.addWidget(self.add_button)
        right_layout.addLayout(preview_layout)

        # Główny układ poziomy
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_column, stretch=3)  # Więcej miejsca dla obrazu
        main_layout.addLayout(right_layout, stretch=1)
        self.setLayout(main_layout)


    def update_frame(self, frame: np.ndarray, detected_faces: list[np.ndarray], recognitions: list[dict]):
        # Konwersja klatki do formatu RGB
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            # Tworzymy QImage
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Skalowanie zachowujące proporcje
            pixmap = QPixmap.fromImage(qt_image)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)

        self._update_face_comparisons(detected_faces, recognitions)

    def _setup_window_geometry(self):
        # Ustawienia bazujące na rozdzielczości ekranu
        screen = QApplication.primaryScreen().availableGeometry()
        margin = 50

        # Oblicz maksymalne rozmiary (90% ekranu z marginesem)
        max_width = min(self._target_width, screen.width() - margin)
        max_height = min(self._target_height, screen.height() - margin)

        self.setMinimumSize(max_width // 2, max_height // 2)
        self.resize(max_width, max_height)

        # Centrowanie okna
        center_point = screen.center()
        self.move(center_point.x() - self.width() // 2,
                  center_point.y() - self.height() // 2)

    def set_video_resolution(self, width: int, height: int):
        self.video_label.setMinimumSize(width // 4, height // 4)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

    def update_video_controls(self, current_frame, total_frames):
        if not hasattr(self, 'video_slider'):
            return

        self.video_slider.setMaximum(total_frames)
        self.video_slider.setValue(current_frame)

        # Aktualizacja czasu tylko jeśli wartości są prawidłowe
        if hasattr(self, 'video_fps') and self.video_fps > 0:
            current_time = current_frame / self.video_fps
            total_time = total_frames / self.video_fps
            self.time_label.setText(
                f"{int(current_time // 60):02d}:{int(current_time % 60):02d}/"
                f"{int(total_time // 60):02d}:{int(total_time % 60):02d}"
            )

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

    @staticmethod
    def _create_face_comparison_widget(cam_face: np.ndarray, ref_face: np.ndarray, label: str) -> QWidget:
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
        """Dodaje kontrolki wideo pod obrazem"""
        # Czyścimy istniejące elementy jeśli są
        while self.video_controls.count():
            item = self.video_controls.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Przycisk pauzy
        self.play_pause_btn = QPushButton("Pauza")
        self.play_pause_btn.setFixedSize(80, 30)

        # Suwak
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)

        # Etykieta czasu
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(100)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Dodajemy kontrolki
        self.video_controls.addWidget(self.play_pause_btn)
        self.video_controls.addWidget(self.video_slider)
        self.video_controls.addWidget(self.time_label)

        # Style dla lepszego wyglądu
        self.video_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                width: 10px;
                background: #555;
                margin: -5px 0;
                border-radius: 5px;
            }
        """)

    def _toggle_play_pause(self):
        # Ta metoda będzie wywoływana przez FaceApp
        pass

    def _on_slider_moved(self, position):
        # Ta metoda będzie wywoływana przez FaceApp
        pass