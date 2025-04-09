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

from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

THRESHOLD = 0.6

class FaceRecognizer:
    def __init__(self, logger: logging.Logger, model_path: str = 'models', known_faces_dir: str = 'my_faces', face_detector_handler=None):
        """
        Initialize face recognition system.

        Args:
            logger: Configured logger instance
            model_path: Path to models directory
            known_faces_dir: Directory with known faces (subdirectories per person)
            face_detector_handler: External face detector passed to this recognizer
        """
        self.logger = logger
        self.model_path = model_path
        self.known_faces_dir = known_faces_dir
        self.known_faces: Dict[str, List[np.ndarray]] = {}
        self.face_cropper = FaceRecImageCropper()
        self.recognition_handler = None
        self.face_detector_handler = face_detector_handler
        self.face_aligner = None

        self._initialize_models()
        self._load_known_faces()

    def _initialize_models(self) -> None:
        """Initialize face recognition and face alignment models"""
        try:
            with open('config/model_conf.yaml') as f:
                model_conf = yaml.load(f, Loader=yaml.FullLoader)

            scene = 'non-mask'

            # --- Face recognition model ---
            rec_category = 'face_recognition'
            rec_name = model_conf[scene][rec_category]
            self.logger.info('Loading face recognition model...')
            rec_loader = FaceRecModelLoader(self.model_path, rec_category, rec_name)
            rec_model, rec_cfg = rec_loader.load_model()
            self.recognition_handler = FaceRecModelHandler(rec_model, 'cpu', rec_cfg)
            self.logger.info('Face recognition model loaded successfully!')

            # --- Face alignment model ---
            align_category = 'face_alignment'
            align_name = model_conf[scene][align_category]
            self.logger.info('Loading face alignment model...')
            align_loader = FaceAlignModelLoader(self.model_path, align_category, align_name)
            align_model, align_cfg = align_loader.load_model()
            self.face_aligner = FaceAlignModelHandler(align_model, 'cpu', align_cfg)
            self.logger.info('Face alignment model loaded successfully!')

        except Exception as e:
            self.logger.error(f'Failed to initialize models: {str(e)}')
            raise

    def initialize_face_detector(self, face_detector_handler):
        """Poprawiona inicjalizacja detektora"""
        self.face_detector_handler = face_detector_handler

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
        if self.face_detector_handler is None or self.face_aligner is None:
            raise RuntimeError("Face detector or face aligner not initialized")

        try:
            # Detekcja twarzy
            detections = self.face_detector_handler.inference_on_image(face_image)
            if detections is None or len(detections) == 0:
                self.logger.warning("No face detected in image")
                return None

            # Weź pierwszą wykrytą twarz
            detection = detections[0]
            x1, y1, x2, y2, _ = list(map(int, detection[:5]))
            bbox = [x1, y1, x2, y2]

            # Obliczanie punktów charakterystycznych (landmarks) do wyrównania twarzy
            landmarks = self.face_aligner.inference_on_image(face_image, bbox)
            if landmarks is None or len(landmarks) == 0:
                self.logger.warning("No landmarks detected for face")
                return None

            # Spłaszczenie listy punktów
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))

            # Przycinanie i wyrównywanie twarzy (alignment)
            cropped_face = self.face_cropper.crop_image_by_mat(face_image, landmarks_list)

            # Generowanie embeddingu
            embedding = self.recognition_handler.inference_on_image(cropped_face)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating face embedding: {str(e)}")
            return None

    def recognize_face(self, face_image: np.ndarray, threshold: float = THRESHOLD) -> Tuple[str, float, Optional[str]]:
        """
        Recognize face from image.

        Args:
            face_image: Face image (numpy array)
            threshold: Similarity threshold for positive recognition

        Returns:
            Tuple: (person_name, similarity_score, reference_image_path)
        """
        if not self.known_faces:
            self.logger.warning("No known faces loaded for recognition")
            return "Unknown", 0.0, None

        try:
            new_embedding = self.generate_embedding(face_image)
            if new_embedding is None:
                return "Unknown", 0.0, None

            best_match = "Unknown"
            best_score = 0.0
            best_reference_image = None

            # Szukaj najlepszego dopasowania i zapamiętaj ścieżkę obrazu
            for person_name in os.listdir(self.known_faces_dir):
                person_dir = os.path.join(self.known_faces_dir, person_name)
                if not os.path.isdir(person_dir):
                    continue

                for img_file in os.listdir(person_dir):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        emb = self.generate_embedding(img)
                        if emb is None:
                            continue

                        similarity = self._cosine_similarity(new_embedding, emb)
                        if similarity > best_score and similarity > threshold:
                            best_score = similarity
                            best_match = person_name
                            best_reference_image = img_path
                    except Exception as e:
                        self.logger.warning(f"Error comparing with {img_path}: {str(e)}")
                        continue

            return best_match, best_score, best_reference_image

        except Exception as e:
            self.logger.error(f"Face recognition error: {str(e)}")
            return "Unknown", 0.0, None

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

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
