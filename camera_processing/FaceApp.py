import cv2
import time
import numpy as np
from collections import OrderedDict
from camera_processing.FaceDetector import FaceDetector
from camera_processing.FaceRecognizer import FaceRecognizer


class FaceApp:
    def __init__(self, logger, ui, video_source):
        self.logger = logger
        self.ui = ui
        self.ui.on_add_face_callback = self._handle_add_face_from_frame
        self.ui.on_prepare_face_crop_callback = self._prepare_face_crop

        self.video_source = video_source

        frame_width, frame_height = self.video_source.get_frame_size()
        self.ui.set_video_resolution(frame_width, frame_height)

        self.detector = FaceDetector(logger)
        self.recognizer = FaceRecognizer(logger, face_detector_handler=self.detector.face_handler)
        self.recognizer.initialize_face_detector(self.detector.face_handler)

        self.detected_faces = []
        self.recognitions = []

        self.last_recognition_time = 0
        self.recognition_interval = 2
        self.next_face_id = 0
        self.face_trackers = OrderedDict()
        self.max_trackers = 10
        self.recognition_memory_time = 5
        self.face_reappear_threshold = 1

        self._processing_scale = 0.5  # Skala przetwarzania
        self._min_face_size = 100    # Minimalny rozmiar twarzy
        self._last_processed = 0     # Ostatnie przetworzenie
        self._processing_interval = 0.1  # 100ms między przetworzeniami

    def update(self):
        """Wywoływane co ~30ms przez QTimer (w GUI)"""
        ret, frame = self.video_source.read()
        if not ret or frame is None:  # frame może być None gdy wideo jest zapauzowane
            return

        frame = self.process_frame(frame)
        self.ui.update_frame(frame, self.detected_faces, self.recognitions)

    def process_frame(self, frame):
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
                time_since_last = time.time() - self.face_trackers[face_id]['last_seen'] if not is_new else float('inf')

                # Czy trzeba rozpoznać ponownie?
                if (is_new or time_since_last > self.face_reappear_threshold or
                        (not is_new and self.face_trackers[face_id]['recognition']['name'] in ['Nieznany',
                                                                                               'Przetwarzanie...'])):
                    recognition = {'name': 'Przetwarzanie...', 'score': 0}
                    needs_recognition = True
                else:
                    recognition = self.face_trackers[face_id]['recognition']
                    needs_recognition = False

                # Wymuszone rozpoznanie
                if needs_recognition or current_time - self.last_recognition_time > self.recognition_interval:
                    name, score, reference_path = self.recognizer.recognize_face(face_img)
                    recognition = {
                        'name': name,
                        'score': score,
                        'reference': reference_path  # <-- to dodajesz
                    }
                    self.last_recognition_time = current_time

                # === RYSOWANIE OBRAZKÓW I NAZW ===
                if recognition['name'] == 'Unknown':
                    color = (0, 0, 255)  # Czerwony
                elif recognition['name'] == 'Przetwarzanie...':
                    color = (255, 165, 0)  # Pomarańczowy
                else:
                    color = (0, 255, 0)  # Zielony

                # Rysowanie obramowania i napisu
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{recognition['name']} ({recognition['score'] * 100:.1f}%)"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Zapamiętaj rozpoznaną twarz
                new_face_trackers[face_id] = {
                    'last_box': box,
                    'recognition': recognition,
                    'last_seen': current_time
                }
                self.recognitions.append(recognition)

        # Wyczyść przeterminowane trackery
        while len(new_face_trackers) > self.max_trackers:
            new_face_trackers.popitem(last=False)

        self.face_trackers = new_face_trackers
        return frame

    def _match_faces_to_trackers(self, detections):
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
                dist = np.linalg.norm(np.array(center) - np.array(last_center))

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

    def _handle_add_face_from_frame(self, frame: np.ndarray, name: str):
        try:
            detections = self.detector.detect_faces(frame)
            if not detections.any():
                self.logger.warning("Nie wykryto twarzy w obrazie.")
                return

            # Zakładamy, że pierwsza wykryta twarz to ta, którą chcemy dodać
            x1, y1, x2, y2 = map(int, detections[0][:4])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                self.logger.warning("Błąd: niewłaściwy rozmiar wykrytej twarzy.")
                return

            cropped_face = frame[y1:y2, x1:x2]

            # Dodaj wyciętą twarz do bazy
            success = self.recognizer.add_new_face(cropped_face, name)
            if success:
                self.logger.info(f"Dodano nową twarz: {name}")
                self.last_recognition_time = 0
            else:
                self.logger.warning(f"Nie udało się dodać twarzy: {name}")

        except Exception as e:
            self.logger.error(f"Błąd przy dodawaniu twarzy: {str(e)}")

    def _prepare_face_crop(self, frame: np.ndarray) -> np.ndarray | None:
        try:
            detections = self.detector.detect_faces(frame)
            if not detections.any():
                return None
            x1, y1, x2, y2 = map(int, detections[0][:4])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return frame[y1:y2, x1:x2]
        except Exception as e:
            self.logger.error(f"Błąd podczas przygotowania podglądu twarzy: {str(e)}")
            return None
