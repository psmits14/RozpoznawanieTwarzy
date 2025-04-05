import os
import cv2
import numpy as np
import yaml
import time
import logging
from typing import Dict, List, Tuple, Optional

from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler


class FaceRecognizer:
    def __init__(self, logger: logging.Logger, model_path: str = 'models', known_faces_dir: str = 'my_faces',  face_detector_handler=None):
        """
        Initialize face recognition system.

        Args:
            logger: Configured logger instance
            model_path: Path to models directory
            known_faces_dir: Directory with known faces (subdirectories per person)
        """
        self.logger = logger
        self.model_path = model_path
        self.known_faces_dir = known_faces_dir
        self.known_faces: Dict[str, List[np.ndarray]] = {}
        self.face_cropper = None
        self.recognition_handler = None
        self.face_detector_handler = face_detector_handler

        self._initialize_models()
        if self.face_detector_handler:
            self.face_cropper = FaceRecImageCropper()
        self._load_known_faces()

    def _initialize_models(self) -> None:
        """Initialize face recognition model"""
        try:
            # Load face recognition model
            with open('config/model_conf.yaml') as f:
                model_conf = yaml.load(f, Loader=yaml.FullLoader)

            scene = 'non-mask'
            model_category = 'face_recognition'
            model_name = model_conf[scene][model_category]

            self.logger.info('Loading face recognition model...')
            model_loader = FaceRecModelLoader(self.model_path, model_category, model_name)
            model, cfg = model_loader.load_model()
            self.recognition_handler = FaceRecModelHandler(model, 'cpu', cfg)
            self.logger.info('Face recognition model loaded successfully!')

        except Exception as e:
            self.logger.error(f'Failed to initialize face recognition model: {str(e)}')
            raise

    def initialize_face_detector(self, face_detector_handler):
        """Poprawiona inicjalizacja detektora"""
        self.face_detector_handler = face_detector_handler
        self.face_cropper = FaceRecImageCropper()  # Przekazanie detektora

    def _load_known_faces(self) -> None:
        """Load known faces from directory structure"""
        if not os.path.exists(self.known_faces_dir):
            self.logger.warning(f"Known faces directory {self.known_faces_dir} does not exist!")
            return

        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                self.known_faces[person_name] = []
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_file)
                        try:
                            embedding = self._generate_embedding_from_image(img_path)
                            if embedding is not None:
                                self.known_faces[person_name].append(embedding)
                                self.logger.debug(f"Loaded face for {person_name} from {img_file}")
                        except Exception as e:
                            self.logger.error(f"Error processing {img_path}: {str(e)}")

        self.logger.info(
            f"Loaded {sum(len(v) for v in self.known_faces.values())} face embeddings for {len(self.known_faces)} known persons")

    def _generate_embedding_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """Generate face embedding from image file"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image {image_path}")

            return self.generate_embedding(img)
        except Exception as e:
            self.logger.error(f"Error generating embedding from {image_path}: {str(e)}")
            return None

    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding from face image (numpy array)"""
        if self.recognition_handler is None:
            raise RuntimeError("Face recognition model not initialized")
        if self.face_detector_handler is None:
            raise RuntimeError("Face detector not initialized")

        try:
            # Detect and align face
            detections = self.face_detector_handler.inference_on_image(face_image)
            if not detections.any():
                self.logger.warning("No face detected in image")
                return None

            # Weź pierwszą wykrytą twarz (możesz zmodyfikować, aby obsłużyć wiele twarzy)
            #face_det = detections[0]

            # Zakładając, że detections zawiera landmarks (punkty charakterystyczne)
            # Format może się różnić w zależności od modelu detekcji
            #landmarks = face_det[5:15]  # Przykład - może być inaczej w Twoim modelu
            #landmarks = np.array(landmarks).reshape(5, 2)  # Konwersja do formatu 5 punktów

            # Teraz przycinamy obraz używając landmarks
            #cropped_face = self.face_cropper.crop_image_by_mat(face_image, landmarks)

            # Weź pierwszą wykrytą twarz
            detection = detections[0]
            x1, y1, x2, y2, confidence = detection

            # Przygotuj bounding box
            bbox = [x1, y1, x2, y2]

            # Przycinanie twarzy na podstawie bounding boxa
            cropped_face = self._crop_face_by_bbox(face_image, bbox)

            # Generate embedding
            embedding = self.recognition_handler.inference_on_image(cropped_face)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating face embedding: {str(e)}")
            return None

    def _crop_face_by_bbox(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """Crop face using bounding box coordinates"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]

        # Zabezpieczenie przed wyjściem poza zakres
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box coordinates")

        cropped = image[y1:y2, x1:x2]

        # Resize do standardowego rozmiaru jeśli potrzebny
        cropped = cv2.resize(cropped, (112, 112))  # standardowy rozmiar dla wielu modeli
        return cropped

    def recognize_face(self, face_image: np.ndarray, threshold: float = 0.6) -> Tuple[str, float]:
        """
        Recognize face from image

        Args:
            face_image: Face image (numpy array)
            threshold: Similarity threshold for positive recognition

        Returns:
            Tuple: (person_name, similarity_score)
        """
        if not self.known_faces:
            self.logger.warning("No known faces loaded for recognition")
            return "Unknown", 0.0

        try:
            # Generate embedding for the new face
            new_embedding = self.generate_embedding(face_image)
            if new_embedding is None:
                return "Unknown", 0.0

            # Compare with known faces
            best_match = "Unknown"
            best_score = 0.0

            for name, embeddings in self.known_faces.items():
                for emb in embeddings:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(new_embedding, emb)
                    if similarity > best_score and similarity > threshold:
                        best_score = similarity
                        best_match = name

            return best_match, best_score
        except Exception as e:
            self.logger.error(f"Face recognition error: {str(e)}")
            return "Unknown", 0.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def add_new_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """
        Add new face to known faces database

        Args:
            face_image: Face image (numpy array)
            person_name: Name to associate with the face

        Returns:
            bool: True if successful
        """
        try:
            # Create directory for person if doesn't exist
            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)

            # Generate unique filename
            timestamp = int(time.time())
            img_path = os.path.join(person_dir, f"{timestamp}.jpg")

            # Save face image
            cv2.imwrite(img_path, face_image)

            # Generate and save embedding
            embedding = self.generate_embedding(face_image)
            if embedding is None:
                raise ValueError("Failed to generate face embedding")

            if person_name not in self.known_faces:
                self.known_faces[person_name] = []
            self.known_faces[person_name].append(embedding)

            self.logger.info(f"Successfully added new face for {person_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add new face: {str(e)}")
            return False