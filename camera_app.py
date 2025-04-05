# camera_app.py

import logging.config
import sys
import warnings
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from camera_processing.FaceCameraApp import FaceCameraApp
from camera_processing.CameraUI import CameraUI  # <-- nowy GUI!

from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def configure_logging():
    logging.config.fileConfig("config/logging.conf")
    return logging.getLogger('api')

if __name__ == "__main__":
    logger = configure_logging()

    app = QApplication(sys.argv)

    gui = CameraUI()
    logic = FaceCameraApp(logger, gui)

    timer = QTimer()
    timer.timeout.connect(logic.update)
    timer.start(30)  # 30ms ~ 33 FPS

    gui.show()
    sys.exit(app.exec())
