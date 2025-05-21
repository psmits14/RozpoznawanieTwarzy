import csv
import cv2
import time
import os
import numpy as np
from collections import OrderedDict
from PyQt6.QtCore import QThread, QUrl
from PyQt6.QtMultimedia import QSoundEffect  # <-- Dźwięk przez Qt

from camera_processing.FaceDetector import FaceDetector
from camera_processing.FaceRecognizer import FaceRecognizer
from camera_processing.DetectionWorker import DetectionWorker


class FaceApp:
    def __init__(self, logger, ui, video_source, recognition_threshold=0.5,  sound_enabled=False):
        self.logger = logger
        self.ui = ui
        self.ui.on_add_face_callback = self._handle_add_face_from_frame
        self.ui.on_prepare_face_crop_callback = self._prepare_face_crop

        self.sound_enabled = sound_enabled
        self.last_alert_sound_time = 0
        self.alert_cooldown = 2  # sekundy

        self.video_source = video_source
        self.recognition_threshold = recognition_threshold
        frame_width, frame_height = self.video_source.get_frame_size()
        self.ui.set_video_resolution(frame_width, frame_height)

        self.detector = FaceDetector(logger)
        self.recognizer = FaceRecognizer(logger, face_detector_handler=self.detector.face_handler)
        self.recognizer.initialize_face_detector(self.detector.face_handler)

        self.detected_faces = []
        self.recognitions = []

        self.last_recognition_time = 0
        self.recognition_interval = 2       # ZMIEN JESLI NIE CHCESZ ZEBY TAK CZESTO MIERZONO ROZPOZNANIE
        self.next_face_id = 0
        self.face_trackers = OrderedDict()
        self.max_trackers = 10
        self.recognition_memory_time = 5
        self.face_reappear_threshold = 1

        self._processing_interval = 0.2
        self._last_processed = 0
        self._last_detections = []
        self._detection_running = False
        self._detection_thread = None

        self.is_video_file = hasattr(video_source, 'get_current_frame_position')
        self.recognition_log = []
        self.output_log_path = None  # zostanie ustawiona później
        # jeśli to plik wideo, ustaw ścieżkę zapisu logu
        if self.is_video_file and hasattr(video_source, 'file_path'):
            video_name = os.path.splitext(os.path.basename(video_source.file_path))[0]
            os.makedirs("test_results", exist_ok=True)
            self.output_log_path = os.path.join("test_results", f"{video_name}_log.csv")
        self._last_logged_time_by_name = {}

        # === SOUND SETUP ===
        self.alert_effect = QSoundEffect()
        self.alert_effect.setSource(QUrl.fromLocalFile("sounds/alert.wav"))
        self.alert_effect.setLoopCount(1)
        self.alert_effect.setVolume(0.9)

    def update(self):
        start = time.time()
        ret, frame = self.video_source.read()
        if not ret or frame is None:
            return

        frame = self.process_frame(frame)
        self.ui.update_frame(frame, self.detected_faces, self.recognitions)
        elapsed = time.time() - start

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        now = time.time()

        if now - self._last_processed >= self._processing_interval and not self._detection_running:
            self._start_async_detection(frame)
            self._last_processed = now

        detections = self._last_detections
        self.detected_faces = []
        current_time = now

        matched_faces = self._match_faces_to_trackers(detections)
        new_face_trackers = OrderedDict()
        self.recognitions = []

        for face_id, det_idx in matched_faces.items():
            box = list(map(int, detections[det_idx][:4]))
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_img = frame[y1:y2, x1:x2]
            self.detected_faces.append(face_img)

            is_new = face_id not in self.face_trackers
            time_since_last = time.time() - self.face_trackers[face_id]['last_seen'] if not is_new else float('inf')

            if (is_new or time_since_last > self.face_reappear_threshold or
                    (not is_new and self.face_trackers[face_id]['recognition']['name'] in ['Nieznany',
                                                                                           'Przetwarzanie...'])):
                recognition = {'name': 'Przetwarzanie...', 'score': 0}
                needs_recognition = True
            else:
                recognition = self.face_trackers[face_id]['recognition']
                needs_recognition = False

            if needs_recognition or current_time - self.last_recognition_time > self.recognition_interval:
                name, score, reference_path = self.recognizer.recognize_face(face_img)
                self.last_recognition_time = current_time

                if score >= self.recognition_threshold:
                    label_name = name
                else:
                    label_name = 'Unknown'
                    reference_path = None
                    score = 0.0

                recognition = {
                    'name': label_name,
                    'score': score,
                    'reference': reference_path
                }

            # Kolory
            if recognition['name'] == 'Unknown':
                color = (0, 0, 255)
                if self.sound_enabled:
                    self._play_unknown_sound()
                    print("ALERTTT")
            elif recognition['name'] == 'Przetwarzanie...':
                color = (255, 165, 0)
            else:
                color = (0, 255, 0)

            label = f"{recognition['name']} ({recognition['score'] * 100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            new_face_trackers[face_id] = {
                'last_box': box,
                'recognition': recognition,
                'last_seen': current_time
            }

            self.recognitions.append(recognition)

            # Zapisz do logu dokładnie to, co wyświetlasz
            if self.is_video_file and recognition['name'] != 'Przetwarzanie...':
                current_frame = self.video_source.get_current_frame_position()
                fps = self.video_source._fps
                current_time_sec = round(current_frame / fps, 2)

                self.recognition_log.append({
                    "time": current_time_sec,
                    "name": recognition['name'],
                    "score": round(recognition['score'] * 100, 1)
                })

        while len(new_face_trackers) > self.max_trackers:
            new_face_trackers.popitem(last=False)

        self.face_trackers = new_face_trackers
        return frame

    def _start_async_detection(self, frame):
        self._detection_running = True
        self._detection_thread = QThread()
        self._detection_worker = DetectionWorker(self.detector, frame.copy())
        self._detection_worker.moveToThread(self._detection_thread)

        self._detection_thread.started.connect(self._detection_worker.run)
        self._detection_worker.finished.connect(self._on_detection_finished)
        self._detection_worker.finished.connect(self._detection_thread.quit)
        self._detection_worker.finished.connect(self._detection_worker.deleteLater)
        self._detection_thread.finished.connect(self._detection_thread.deleteLater)

        self._detection_thread.start()

    def _on_detection_finished(self, detections):
        self._last_detections = detections
        self._detection_running = False

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

            x1, y1, x2, y2 = map(int, detections[0][:4])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                self.logger.warning("Błąd: niewłaściwy rozmiar wykrytej twarzy.")
                return

            cropped_face = frame[y1:y2, x1:x2]
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

    def save_recognition_log(self):
        if not self.is_video_file or not self.recognition_log or not self.output_log_path:
            return

        try:
            with open(self.output_log_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["time", "name", "score"])
                writer.writeheader()
                for row in self.recognition_log:
                    writer.writerow(row)
            self.logger.info(f"Zapisano log rozpoznań do pliku: {self.output_log_path}")
        except Exception as e:
            self.logger.error(f"Błąd zapisu logu rozpoznań: {str(e)}")


    def _play_unknown_sound(self):
        now = time.time()
        if now - self.last_alert_sound_time < self.alert_cooldown:
            return

        if self.alert_effect.isPlaying():
            return

        self.logger.info("[DŹWIĘK] Odtwarzanie alertu...")
        self.alert_effect.play()
        self.last_alert_sound_time = now






