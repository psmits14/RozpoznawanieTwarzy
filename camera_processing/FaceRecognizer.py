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

THRESHOLD = 0.5     # Próg podobieństwa do rozpoznawania twarzy

class FaceRecognizer:
    def __init__(self, logger: logging.Logger, model_path: str = 'models', known_faces_dir: str = 'my_faces', face_detector_handler=None):
        """
        Konstruktor klasy do rozpoznawania twarzy.
        Inicjalizuje ścieżki, modele, loggera oraz ładuje znane twarze.
        """
        self.logger = logger
        self.model_path = model_path
        self.known_faces_dir = known_faces_dir
        self.known_faces: Dict[str, List[Tuple[np.ndarray, str]]] = {}
        self.face_cropper = FaceRecImageCropper()
        self.recognition_handler = None
        self.face_detector_handler = face_detector_handler
        self.face_aligner = None

        self._initialize_models()
        self._load_known_faces()

    def _initialize_models(self) -> None:
        """
        Inicjalizacja modeli do rozpoznawania i wyrównywania twarzy na podstawie konfiguracji YAML.
        """
        try:
            with open('config/model_conf.yaml') as f:
                model_conf = yaml.load(f, Loader=yaml.FullLoader)

            scene = 'non-mask'

            # Załaduj model rozpoznawania twarzy
            rec_category = 'face_recognition'
            rec_name = model_conf[scene][rec_category]
            self.logger.info('Loading face recognition model...')
            rec_loader = FaceRecModelLoader(self.model_path, rec_category, rec_name)
            rec_model, rec_cfg = rec_loader.load_model()
            self.recognition_handler = FaceRecModelHandler(rec_model, 'cpu', rec_cfg)
            self.logger.info('Face recognition model loaded successfully!')

            # Załaduj model do wyrównywania twarzy (landmarki)
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
        """Zewnętrzna inicjalizacja handlera detekcji twarzy"""
        self.face_detector_handler = face_detector_handler

    def _load_known_faces(self) -> None:
        """
        Ładuje znane twarze z katalogu i generuje dla nich embeddingi.
        """
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
                                self.known_faces[person_name].append((embedding, img_path))
                                self.logger.debug(f"Loaded face for {person_name} from {img_file}")
                        except Exception as e:
                            self.logger.error(f"Error processing {img_path}: {str(e)}")

        total_embeddings = sum(len(v) for v in self.known_faces.values())
        self.logger.info(f"Loaded {total_embeddings} face embeddings for {len(self.known_faces)} known persons")

    def _generate_embedding_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """Wczytuje obraz z dysku i generuje embedding"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image {image_path}")
            return self.generate_embedding(img)
        except Exception as e:
            self.logger.error(f"Error generating embedding from {image_path}: {str(e)}")
            return None

    def generate_embedding(self, face_image: np.ndarray, return_landmarks=False) -> Optional[tuple]:
        """
         Generuje embedding twarzy z podanego obrazu.
        """
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

            # Wyrównanie twarzy (landmarki)
            landmarks = self.face_aligner.inference_on_image(face_image, bbox)
            if landmarks is None or len(landmarks) == 0:
                self.logger.warning("No landmarks detected for face")
                return None

            # Przygotowanie listy landmarków
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))

            # Kadrowanie twarzy na podstawie landmarków
            cropped_face = self.face_cropper.crop_image_by_mat(face_image, landmarks_list)

            # Generowanie embeddingu
            embedding = self.recognition_handler.inference_on_image(cropped_face)

            if return_landmarks:
                return embedding, landmarks
            else:
                return embedding

        except Exception as e:
            self.logger.error(f"Error generating face embedding: {str(e)}")
            return None

    def recognize_face(self, face_image: np.ndarray, threshold: float = THRESHOLD) -> Tuple[str, float, Optional[str]]:
        """
        Rozpoznaje twarz na podstawie znanych embeddingów (wektorów cech).
        Zwraca:
            - nazwę osoby (jeśli została rozpoznana) lub "Unknown",
            - wartość podobieństwa (Similarity Score),
            - ścieżkę do pasującego obrazu wzorcowego (jeśli istnieje).
        """

        # Jeśli nie ma żadnych zapisanych twarzy, nie można rozpoznać – zwracamy "Unknown"
        if not self.known_faces:
            self.logger.warning("No known faces loaded for recognition")
            return "Unknown", 0.0, None

        try:
            # Generujemy embedding oraz punkty charakterystyczne (landmarki)
            result = self.generate_embedding(face_image, return_landmarks=True)
            if result is None:
                return "Unknown", 0.0, None

            new_embedding, landmarks = result

            # Próba wykrycia okularów
            if self._has_glasses(face_image, landmarks):
                self.logger.debug("[GLASSES] Okulary wykryte.")
            else:
                self.logger.debug("[GLASSES] Brak okularów.")

            # Inicjalizacja zmiennych dla najlepszego dopasowania
            best_match = "Unknown"
            best_score = 0.0
            best_reference_image = None

            # Iteracja po wszystkich znanych osobach i ich embeddingach
            for person_name, embeddings in self.known_faces.items():
                for emb, img_path in embeddings:
                    # Obliczamy podobieństwo pomiędzy nowym embeddingiem a wzorcem
                    similarity = self._cosine_similarity(new_embedding, emb)

                    # Jeśli znaleziono lepsze dopasowanie – zapamiętujemy
                    if similarity > best_score:
                        best_score = similarity
                        best_match = person_name
                        best_reference_image = img_path

            # Jeśli najlepszy wynik nie przekracza progu – uznajemy, że osoba jest nieznana
            if best_score < THRESHOLD:
                best_match = "Unknown"
                best_reference_image = None

            return best_match, best_score, best_reference_image

        except Exception as e:
            self.logger.error(f"Face recognition error: {str(e)}")
            return "Unknown", 0.0, None


        except Exception as e:
            self.logger.error(f"Face recognition error: {str(e)}")
            return "Unknown", 0.0, None

    def add_new_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """
        Dodaje nową twarz do bazy znanych osób:
        - zapisuje zdjęcie twarzy w odpowiednim folderze,
        - generuje embedding (wektor cech),
        - dodaje dane do słownika znanych twarzy.

        Zwraca True w przypadku sukcesu, False w przypadku błędu.
        """
        try:
            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            timestamp = int(time.time())
            img_path = os.path.join(person_dir, f"{timestamp}.jpg")
            cv2.imwrite(img_path, face_image)

            # Generujemy embedding twarzy – wektor cech potrzebny do porównywania
            embedding = self.generate_embedding(face_image)
            if embedding is None:
                raise ValueError("Failed to generate face embedding")

            # Inicjalizacja listy embeddingów dla osoby, jeśli to jej pierwsze wystąpienie
            if person_name not in self.known_faces:
                self.known_faces[person_name] = []

            # Dodajemy embedding i ścieżkę do zdjęcia do listy znanych twarzy
            self.known_faces[person_name].append((embedding, img_path))

            self.logger.info(f"Successfully added new face for {person_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add new face: {str(e)}")
            return False

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """ Oblicza podobieństwo cosinusowe między dwoma wektorami."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _has_glasses(self, img: np.ndarray, landmarks: np.ndarray) -> bool:
        """ Próbuje wykryć okulary na podstawie zmienności jasności w obszarze oczu."""
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape((-1, 2)).astype(np.int32)

        # Wydziel punkty odpowiadające lewemu i prawemu oku
        left_eye = landmarks[66:74]
        right_eye = landmarks[75:83]

        def region_stats(points):
            """
            Dla podanych punktów (np. obszaru oka) oblicz:
            - odchylenie standardowe jasności (std),
            - średnią jasność (mean)
            """
            if len(points) == 0:
                return 0, 0

            # Wyznacz prostokąt ograniczający obszar punktów
            rect = cv2.boundingRect(np.array(points, dtype=np.int32))
            x, y, w, h = rect

            # Wyodrębnij region obrazu odpowiadający temu prostokątowi
            roi = img[y:y + h, x:x + w]
            if roi.size == 0:
                return 0, 0

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Zwróć odchylenie standardowe i średnią jasność
            return np.std(gray), np.mean(gray)

        std_l, mean_l = region_stats(left_eye)
        std_r, mean_r = region_stats(right_eye)

        mean_std = (std_l + std_r) / 2
        mean_mean = (mean_l + mean_r) / 2

        self.logger.debug(f"[GLASSES-CHECK] std: {mean_std:.2f}, mean: {mean_mean:.2f}")

        # Jeśli odchylenie standardowe jest wysokie – możliwe, że są okulary
        if mean_std > 30:
            self.logger.debug("[GLASSES-CHECK] ✅ WYKRYTO OKULARY")
            return True
        else:
            self.logger.debug("[GLASSES-CHECK] ❌ Brak okularów")
            return False