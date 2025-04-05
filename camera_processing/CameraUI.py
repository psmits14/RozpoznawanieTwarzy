from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import cv2
import numpy as np


class CameraUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikacja Kamery - Detekcja i Rozpoznawanie Twarzy")
        self.setMinimumSize(1200, 600)

        # Stan aplikacji
        self.current_frame = None
        self.pending_face_crop = None
        self.on_add_face_callback = None
        self.on_prepare_face_crop_callback = None

        self._setup_ui()

    def _setup_ui(self):
        # Obraz z kamery (rozmiar zostanie ustawiony dynamicznie)
        self.video_label = QLabel("Obraz z kamery")
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Lista twarzy
        self.face_list = QListWidget()
        self.face_list.setFixedWidth(400)

        # Pole do wpisania imienia
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Wpisz imię i naciśnij ENTER")
        self.name_input.returnPressed.connect(self._submit_face_name)
        self.name_input.hide()

        # Przycisk dodania nowej twarzy
        self.add_button = QPushButton("Dodaj nową twarz")
        self.add_button.clicked.connect(self._on_add_clicked)

        # Podgląd twarzy
        self.face_preview = QLabel("Podgląd twarzy")
        self.face_preview.setFixedSize(120, 120)
        self.face_preview.setStyleSheet("border: 1px solid gray; background: #eee;")
        self.face_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_preview.setVisible(False)

        # Przyciski zatwierdź/anuluj
        self.confirm_button = QPushButton("Zatwierdź dodanie")
        self.confirm_button.clicked.connect(self._confirm_add_face)
        self.confirm_button.setVisible(False)

        self.cancel_button = QPushButton("Anuluj")
        self.cancel_button.clicked.connect(self._cancel_add_face)
        self.cancel_button.setVisible(False)

        button_row = QHBoxLayout()
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.confirm_button)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.face_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        preview_layout.addLayout(button_row)

        # Panel boczny
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Rozpoznane twarze:"))
        right_layout.addWidget(self.face_list)
        right_layout.addWidget(self.name_input)
        right_layout.addWidget(self.add_button)
        right_layout.addLayout(preview_layout)

        # Główny układ
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def set_video_resolution(self, width: int, height: int):
        """Dopasowuje QLabel do rozmiaru klatki z kamery"""
        self.video_label.setFixedSize(width, height)

    def update_frame(self, frame: np.ndarray,
                     detected_faces: list[np.ndarray],
                     recognitions: list[dict]):
        """Aktualizuje obraz z kamery i listę rozpoznanych twarzy"""
        self.current_frame = frame.copy()
        self._update_video_display(frame)
        self._update_face_list(detected_faces, recognitions)

    def _update_video_display(self, frame: np.ndarray):
        """Wyświetla obraz z kamery bez skalowania"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def _update_face_list(self, faces, recognitions):
        """Aktualizuje listę twarzy po prawej stronie"""
        self.face_list.clear()
        for i, (face, rec) in enumerate(zip(faces, recognitions)):
            name = rec.get("name", "Nieznany")
            score = rec.get("score", 0)
            label = f"{i + 1}. {name} - {score * 100:.1f}%"
            self.face_list.addItem(QListWidgetItem(label))

    def _on_add_clicked(self):
        """Obsługa kliknięcia przycisku 'Dodaj nową twarz'"""
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
        """Wyświetla miniaturkę wyciętej twarzy przed zatwierdzeniem"""
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
