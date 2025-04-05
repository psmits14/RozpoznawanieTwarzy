# python camera_app.py

import logging.config
from camera_processing.FaceCameraApp import FaceCameraApp

def configure_logging():
    logging.config.fileConfig("config/logging.conf")
    return logging.getLogger('api')

if __name__ == "__main__":
    logger = configure_logging()
    try:
        app = FaceCameraApp(logger)
        app.run()
    except Exception as e:
        logger.error(f"Aplikacja zakończyła się błędem: {str(e)}")
        raise