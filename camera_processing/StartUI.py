from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QSlider, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt
import cv2


class StartUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wybór źródła obrazu")
        self.setFixedSize(400, 350)
        self.sound_enabled = False          # Flaga określająca, czy dźwięk jest włączony
        self.on_camera_selected = None
        self.on_video_selected = None

        self._setup_ui()

    def _setup_ui(self):
        """Główny pionowy layout z wyśrodkowaniem i odstępami między elementami"""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(25)

        # Nazwa okna
        title = QLabel("Wybierz źródło obrazu")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        # Przyciski wyboru
        camera_btn = QPushButton("Kamera")
        camera_btn.setFixedSize(200, 50)
        camera_btn.clicked.connect(self._handle_camera_click)

        video_btn = QPushButton("Plik wideo")
        video_btn.setFixedSize(200, 50)
        video_btn.clicked.connect(self._handle_video_click)

        for btn in (camera_btn, video_btn):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #444;
                    color: white;
                    font-size: 16px;
                    border-radius: 6px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
            """)

        # Próg rozpoznania
        self.threshold_label = QLabel("Próg rozpoznania: 50%")
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.threshold_label.setStyleSheet("font-size: 14px;")

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #ccc;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #666;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #0078d7;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #888;
                border-radius: 3px;
            }
            QSlider::tick-mark:horizontal {
                background: white;
                height: 6px;
                width: 1px;
            }
        """)
        # Przycisk włączania/wyłączania dźwięku
        self.sound_button = QPushButton()
        self._update_sound_button_text()
        self.sound_button.setFixedSize(200, 30)
        self.sound_button.clicked.connect(self._toggle_sound)
        self.sound_button.setStyleSheet("font-size: 13px;")

        # Układ główny
        layout.addWidget(title)
        layout.addWidget(camera_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(video_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(20)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_slider)
        layout.addSpacing(10)
        layout.addWidget(self.sound_button, alignment=Qt.AlignmentFlag.AlignCenter)


        self.setLayout(layout)

    def _handle_camera_click(self):
        """Próba otwarcia kamery pod indeksem 0 (domyślna kamera)"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Błąd", "Nie można uruchomić kamery.")
            return
        cap.release()

        if self.on_camera_selected:
            self.on_camera_selected()

    def _handle_video_click(self):
        """Otwarcie okna dialogowego do wyboru pliku wideo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz plik wideo",
            "",
            "Pliki wideo (*.mp4 *.avi *.mov);;Wszystkie pliki (*)"
        )

        if not file_path:
            return

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Błąd", "Nie można otworzyć pliku wideo. Sprawdź format pliku.")
            return
        cap.release()

        if self.on_video_selected:
            self.on_video_selected(file_path)

    def _on_threshold_changed(self, value):
        """Aktualizacja tekstu etykiety progu rozpoznania podczas przesuwania suwaka"""
        self.threshold_label.setText(f"Próg rozpoznania: {value}%")

    def get_threshold_value(self) -> float:
        """Zwraca wartość progu rozpoznania jako liczba zmiennoprzecinkowa 0.0-1.0"""
        return self.threshold_slider.value() / 100.0

    def _toggle_sound(self):
        """Przełączanie stanu dźwięku (włącz/wyłącz)"""
        self.sound_enabled = not self.sound_enabled
        self._update_sound_button_text()

    def _update_sound_button_text(self):
        """Ustawia tekst przycisku dźwięku w zależności od stanu"""
        status = "Włączony" if self.sound_enabled else "Wyłączony"
        self.sound_button.setText(f"Dźwięk: {status}")

    def is_sound_enabled(self) -> bool:
        """Zwraca aktualny stan dźwięku (czy jest włączony sygnał dżwiękowy dla nierozpoznanych twarzy)"""
        return self.sound_enabled

