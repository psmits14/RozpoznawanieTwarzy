# camera_processing/StartUI.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
import cv2


class StartUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wybór źródła obrazu")
        self.setFixedSize(400, 300)

        self.on_camera_selected = None
        self.on_video_selected = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)

        title = QLabel("Wybierz źródło obrazu")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        camera_btn = QPushButton("Kamera")
        camera_btn.setFixedSize(200, 50)
        camera_btn.clicked.connect(self._handle_camera_click)

        video_btn = QPushButton("Plik wideo")
        video_btn.setFixedSize(200, 50)
        video_btn.clicked.connect(self._handle_video_click)

        layout.addWidget(title)
        layout.addWidget(camera_btn)
        layout.addWidget(video_btn)

        self.setLayout(layout)

    def _handle_camera_click(self):
        # Sprawdź czy kamera jest dostępna
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(
                self,
                "Błąd",
                "Nie można uruchomić kamery. Sprawdź czy jest podłączona."
            )
            return
        cap.release()

        if self.on_camera_selected:
            self.on_camera_selected()

    def _handle_video_click(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz plik wideo",
            "",
            "Pliki wideo (*.mp4 *.avi *.mov);;Wszystkie pliki (*)"
        )

        if not file_path:
            return

        # Sprawdź czy plik wideo można otworzyć
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            QMessageBox.critical(
                self,
                "Błąd",
                "Nie można otworzyć pliku wideo. Sprawdź format pliku."
            )
            return
        cap.release()

        if self.on_video_selected:
            self.on_video_selected(file_path)