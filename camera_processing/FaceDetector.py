import cv2
import yaml
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler


class FaceDetector:
    def __init__(self, logger, model_path='models'):
        self.logger = logger
        self.model_path = model_path
        self.face_handler = None
        self._initialize_detector()

    def _initialize_detector(self):
        """Inicjalizacja modelu detekcji twarzy"""
        with open('config/model_conf.yaml') as f:
            model_conf = yaml.load(f, Loader=yaml.FullLoader)

        scene = 'non-mask'
        model_category = 'face_detection'
        model_name = model_conf[scene][model_category]

        self.logger.info('Ładowanie modelu detekcji twarzy...')
        try:
            model_loader = FaceDetModelLoader(self.model_path, model_category, model_name)
            model, cfg = model_loader.load_model()
            self.face_handler = FaceDetModelHandler(model, 'cpu', cfg)
            self.logger.info('Model detekcji twarzy załadowany pomyślnie!')
        except Exception as e:
            self.logger.error('Błąd ładowania modelu detekcji twarzy!')
            raise

    def detect_faces(self, frame):
        """Wykrywa twarze na podanej klatce"""
        if self.face_handler is None:
            raise RuntimeError("Detektor twarzy nie został zainicjalizowany")

        try:
            detections = self.face_handler.inference_on_image(frame)
            return detections
        except Exception as e:
            self.logger.error(f'Błąd detekcji twarzy: {str(e)}')
            raise