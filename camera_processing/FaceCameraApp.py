import cv2
import time
import logging
import numpy as np
from collections import OrderedDict
from camera_processing.FaceDetector import FaceDetector
from camera_processing.FaceRecognizer import FaceRecognizer
from camera_processing.CameraUI import CameraUI


class FaceCameraApp:
    def __init__(self, logger):
        self.logger = logger
        self.camera = self._initialize_camera()
        self.detector = FaceDetector(logger)
        self.recognizer = FaceRecognizer(logger, face_detector_handler=self.detector.face_handler)
        self.recognizer.initialize_face_detector(self.detector.face_handler)  # Udostępniamy detektor do rozpoznawania
        self.ui = CameraUI()

        self.detected_faces = []
        self.recognitions = []
        self.last_recognition_time = 0
        self.recognition_interval = 2  # sekundy między rozpoznaniami
        self.next_face_id = 0
        self.face_trackers = OrderedDict()  # Używamy OrderedDict dla lepszego zarządzania
        self.max_trackers = 10  # Maksymalna liczba śledzonych twarzy
        self.recognition_memory_time = 5  # Czas pamiętania rozpoznania (sekundy)
        self.face_reappear_threshold = 1  # Czas po którym uznamy, że to nowa twarz

    def _initialize_camera(self):
        """Inicjalizacja kamery"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Nie można uruchomić kamery.")
            raise RuntimeError("Nie można uruchomić kamery")
        return cap

    def _match_faces_to_trackers(self, detections):
        """Dopasowuje nowe detekcje do istniejących trackerów"""
        current_boxes = [list(map(int, det[:4])) for det in detections]
        matched = {}
        used_trackers = set()

        for i, box in enumerate(current_boxes):
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            best_match = None
            best_dist = float('inf')

            for face_id, tracker in self.face_trackers.items():
                if face_id in used_trackers:
                    continue

                last_box = tracker['last_box']
                last_center = ((last_box[0] + last_box[2]) // 2, (last_box[1] + last_box[3]) // 2)
                dist = np.sqrt((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2)

                time_since_last = time.time() - tracker['last_seen']
                max_dist = 100 if time_since_last < self.face_reappear_threshold else 50

                if dist < max_dist and dist < best_dist:
                    best_match = face_id
                    best_dist = dist

            if best_match is not None:
                matched[best_match] = i
                used_trackers.add(best_match)
            else:
                new_id = self.next_face_id
                while new_id in self.face_trackers:
                    new_id += 1
                self.next_face_id = new_id + 1
                matched[new_id] = i

        current_time = time.time()
        to_delete = [face_id for face_id, tracker in self.face_trackers.items()
                     if current_time - tracker['last_seen'] > self.recognition_memory_time]

        for face_id in to_delete:
            del self.face_trackers[face_id]

        return matched

    def process_frame(self, frame):
        """Przetwarza pojedynczą klatkę"""
        frame = cv2.flip(frame, 1)
        detections = self.detector.detect_faces(frame)
        self.detected_faces = []
        current_time = time.time()

        matched_faces = self._match_faces_to_trackers(detections)
        new_face_trackers = OrderedDict()
        self.recognitions = []

        for face_id, det_idx in matched_faces.items():
            box = list(map(int, detections[det_idx][:4]))
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 > x1 and y2 > y1:
                face_img = frame[y1:y2, x1:x2]
                self.detected_faces.append(face_img)

                is_new = face_id not in self.face_trackers
                time_since_last = current_time - self.face_trackers[face_id]['last_seen'] if not is_new else float(
                    'inf')

                # Resetuj rozpoznanie jeśli:
                # 1. To nowa twarz
                # 2. Twarz była nieobecna dłużej niż próg
                # 3. Poprzednie rozpoznanie było nieudane
                if (is_new or
                        time_since_last > self.face_reappear_threshold or
                        (not is_new and self.face_trackers[face_id]['recognition']['name'] in ['Nieznany',
                                                                                               'Przetwarzanie...'])):
                    recognition = {'name': 'Przetwarzanie...', 'score': 0}
                    needs_recognition = True
                else:
                    recognition = self.face_trackers[face_id]['recognition']
                    needs_recognition = False

                # Wymuś rozpoznanie jeśli:
                # 1. Jest potrzebne nowe rozpoznanie
                # 2. Minął interwał rozpoznawania
                if needs_recognition or current_time - self.last_recognition_time > self.recognition_interval:
                    name, score = self.recognizer.recognize_face(face_img)
                    recognition = {'name': name, 'score': score}
                    self.last_recognition_time = current_time

                # Kolor ramki
                if recognition['name'] == 'Unknown':
                    color = (0, 0, 255)  # Czerwony
                elif recognition['name'] == 'Processing...':
                    color = (255, 165, 0)  # Pomarańczowy
                else:
                    color = (0, 255, 0)  # Zielony

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{recognition['name']} ({recognition['score'] * 100:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                new_face_trackers[face_id] = {
                    'last_box': box,
                    'recognition': recognition,
                    'last_seen': current_time
                }
                self.recognitions.append(recognition)

        while len(new_face_trackers) > self.max_trackers:
            new_face_trackers.popitem(last=False)

        self.face_trackers = new_face_trackers
        return frame

    def run(self):
        """Główna pętla aplikacji"""
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Nie można odczytać obrazu z kamery.")
                    break

                processed_frame = self.process_frame(frame)
                self.ui.display_frame(processed_frame, self.detected_faces, self.recognitions)

                # Obsługa zdarzeń klawiatury
                action = self.ui.handle_key_events()
                if action and action['action'] == 'add_face':
                    face_idx = action['face_idx']
                    if 0 <= face_idx < len(self.detected_faces):
                        success = self.recognizer.add_new_face(
                            self.detected_faces[face_idx],
                            action['name']
                        )
                        if success:
                            self.logger.info(f"Dodano nową twarz: {action['name']}")
                            # Odśwież rozpoznania po dodaniu nowej twarzy
                            self.last_recognition_time = 0

                if self.ui.should_close():
                    break

        finally:
            self.camera.release()
            cv2.destroyAllWindows()