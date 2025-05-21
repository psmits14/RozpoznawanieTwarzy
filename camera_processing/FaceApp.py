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
        self.ui = ui                                                        # Interfejs użytkownika
        self.ui.on_add_face_callback = self._handle_add_face_from_frame     # Callback do dodawania twarzy
        self.ui.on_prepare_face_crop_callback = self._prepare_face_crop     # Callback do przygotowania podglądu twarzy

        self.sound_enabled = sound_enabled      # Czy umożliwiono alerty dżwiękowe
        self.last_alert_sound_time = 0          # Czas ostatniego odtworzenia alertu
        self.alert_cooldown = 2                 # Minimalny czas (s) między alertami dźwiękowymi

        self.video_source = video_source                                # Źródło obrazu wideo
        self.recognition_threshold = recognition_threshold              # Próg pewności rozpoznania twarzy
        frame_width, frame_height = self.video_source.get_frame_size()
        self.ui.set_video_resolution(frame_width, frame_height)         # Ustawienie rozmiaru obrazu w UI

        # Inicjalizacja detektora i rozpoznawania twarzy
        self.detector = FaceDetector(logger)
        self.recognizer = FaceRecognizer(logger, face_detector_handler=self.detector.face_handler)
        self.recognizer.initialize_face_detector(self.detector.face_handler)

        # Dane wykrytych twarzy i rozpoznań
        self.detected_faces = []            # Lista aktualnie wykrytych twarzy
        self.recognitions = []              # Lista wyników rozpoznania twarzy

        # Parametry rozpoznawania twarzy
        self.last_recognition_time = 0      # Czas ostatniego rozpoznania
        self.recognition_interval = 2       # Odstęp czasowy między rozpoznaniami
        self.next_face_id = 0               # Id kolejnej twarzy
        self.face_trackers = OrderedDict()  # Słownik śledzonych twarzy: ID -> dane
        self.max_trackers = 10              # Maksymalna liczba aktywnych śledzonych twarzy
        self.recognition_memory_time = 5    # Czas (s), po którym nieaktywna twarz jest usuwana
        self.face_reappear_threshold = 1    # Próg (s) pojawiania się twarzy

        # Parametry przetwarzania detekcji
        self._processing_interval = 0.2     # Odstęp między kolejnymi detekcjami (s)
        self._last_processed = 0            # Czas ostatniej detekcji
        self._last_detections = []          # Ostatnio wykryte twarze
        self._detection_running = False     # Czy detekcja jest w toku
        self._detection_thread = None       # Wątek, w którym działa detekcja

        # Jeśli źródłem jest plik wideo – ustaw flagę i nazwę pliku do logowania
        self.is_video_file = hasattr(video_source, 'get_current_frame_position')
        self.recognition_log = []           # Log wyników rozpoznania (dla CSV)
        self.output_log_path = None         # Ścieżka do pliku logu wyników rozpoznania
        # jeśli to plik wideo, ustaw ścieżkę zapisu logu
        if self.is_video_file and hasattr(video_source, 'file_path'):
            video_name = os.path.splitext(os.path.basename(video_source.file_path))[0]
            os.makedirs("test_results", exist_ok=True)
            self.output_log_path = os.path.join("test_results", f"{video_name}_log.csv")
        self._last_logged_time_by_name = {}

        # Inicjalizacja dźwięku alertu
        self.alert_effect = QSoundEffect()
        self.alert_effect.setSource(QUrl.fromLocalFile("sounds/alert.wav")) # Ścieżka do pliku dźwiękowego
        self.alert_effect.setLoopCount(1)                                   # Odtwarzanie tylko raz
        self.alert_effect.setVolume(0.9)                                    # Głośność

    def update(self):
        """
        Główna metoda aktualizująca obraz w pętli głównej aplikacji.
        - Pobiera nową klatkę z kamery lub pliku wideo,
        - Przetwarza ją (detekcja twarzy, rozpoznawanie itp.),
        - Aktualizuje interfejs graficzny.
        """
        ret, frame = self.video_source.read()

        if not ret or frame is None:
            return

        frame = self.process_frame(frame)
        self.ui.update_frame(frame, self.detected_faces, self.recognitions)

    def process_frame(self, frame):
        """
        Przetwarza pojedynczą klatkę obrazu:
        - odbicie lustrzane,
        - detekcja twarzy (asynchroniczna),
        - przypisanie wykryć do śledzonych twarzy,
        - rozpoznanie twarzy,
        - rysowanie ramek i etykiet,
        - logowanie wyników (dla nagrania).
        """
        # Odbicie lustrzane obrazu – lepsze UX dla kamery (jak w lustrze)
        frame = cv2.flip(frame, 1)
        now = time.time()

        # Uruchamiamy detekcję co określony interwał czasu (aby nie przeciążać systemu)
        if now - self._last_processed >= self._processing_interval and not self._detection_running:
            self._start_async_detection(frame)
            self._last_processed = now

        # Pobieramy ostatnie wykryte twarze (asynchronicznie zaktualizowane)
        detections = self._last_detections
        self.detected_faces = []  # Reset listy aktualnie widocznych twarzy
        current_time = now

        # Dopasowanie detekcji do istniejących trackerów (śledzenie twarzy)
        matched_faces = self._match_faces_to_trackers(detections)
        new_face_trackers = OrderedDict()  # Trackery do aktualizacji
        self.recognitions = []  # Lista rozpoznań w bieżącej klatce

        for face_id, det_idx in matched_faces.items():
            # Wyciągamy współrzędne wykrytej twarzy
            box = list(map(int, detections[det_idx][:4]))
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Pomijamy nieprawidłowe detekcje
            if x2 <= x1 or y2 <= y1:
                continue

            # Wycinamy twarz z obrazu
            face_img = frame[y1:y2, x1:x2]
            self.detected_faces.append(face_img)

            # Sprawdzamy, czy to nowa twarz
            is_new = face_id not in self.face_trackers
            time_since_last = time.time() - self.face_trackers[face_id]['last_seen'] if not is_new else float('inf')

            # Czy konieczne jest ponowne rozpoznanie twarzy?
            if (is_new or time_since_last > self.face_reappear_threshold or
                    (not is_new and self.face_trackers[face_id]['recognition']['name'] in ['Nieznany',
                                                                                           'Przetwarzanie...'])):
                recognition = {'name': 'Przetwarzanie...', 'score': 0}
                needs_recognition = True
            else:
                recognition = self.face_trackers[face_id]['recognition']
                needs_recognition = False

            # Jeśli minął interwał rozpoznawania lub twarz jest nowa - rozpoznaj
            if needs_recognition or current_time - self.last_recognition_time > self.recognition_interval:
                name, score, reference_path = self.recognizer.recognize_face(face_img)
                self.last_recognition_time = current_time

                # Sprawdzenie progu dopasowania
                if score >= self.recognition_threshold:
                    label_name = name
                else:
                    label_name = 'Unknown'
                    reference_path = None
                    score = 0.0

                # Zapisz dane rozpoznania
                recognition = {
                    'name': label_name,
                    'score': score,
                    'reference': reference_path
                }

            # Ustaw kolor ramki w zależności od statusu rozpoznania
            if recognition['name'] == 'Unknown':
                color = (0, 0, 255)  # czerwony
                if self.sound_enabled:
                    self._play_unknown_sound()
            elif recognition['name'] == 'Przetwarzanie...':
                color = (255, 165, 0)  # pomarańczowy
            else:
                color = (0, 255, 0)  # zielony

            # Napis z nazwą i procentową wartością dopasowania
            label = f"{recognition['name']} ({recognition['score'] * 100:.1f}%)"

            # Rysowanie ramki i etykiety na obrazie
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Zapisanie/odświeżenie trackera dla twarzy
            new_face_trackers[face_id] = {
                'last_box': box,
                'recognition': recognition,
                'last_seen': current_time
            }

            # Dodanie do listy rozpoznań
            self.recognitions.append(recognition)

            # Logowanie rozpoznania (jeśli analizujemy plik wideo)
            if self.is_video_file and recognition['name'] != 'Przetwarzanie...':
                current_frame = self.video_source.get_current_frame_position()
                fps = self.video_source._fps
                current_time_sec = round(current_frame / fps, 2)

                self.recognition_log.append({
                    "time": current_time_sec,
                    "name": recognition['name'],
                    "score": round(recognition['score'] * 100, 1)
                })

        # Ograniczamy liczbę trackerów (FIFO) – usuwamy najstarsze, jeśli za dużo
        while len(new_face_trackers) > self.max_trackers:
            new_face_trackers.popitem(last=False)

        # Aktualizacja stanu trackerów
        self.face_trackers = new_face_trackers

        # Zwracamy zmodyfikowaną klatkę (z narysowanymi ramkami)
        return frame

    def _start_async_detection(self, frame):
        """
        Asynchroniczne uruchomienie detekcji twarzy w osobnym wątku (QThread).
        Dzięki temu główny wątek GUI pozostaje responsywny (nie zawiesza się).
        """
        # Ustawiamy flagę informującą, że detekcja jest w toku
        self._detection_running = True

        # Tworzymy nowy wątek Qt
        self._detection_thread = QThread()

        # Tworzymy instancję pracownika detekcji (DetectionWorker), przekazując mu detektor i kopię klatki do analizy
        self._detection_worker = DetectionWorker(self.detector, frame.copy())

        # Przenosimy obiekt detekcji do nowego wątku – będzie tam działać
        self._detection_worker.moveToThread(self._detection_thread)

        # Łączymy sygnały: Kiedy wątek się uruchomi odpal metodę `run` w DetectionWorker
        self._detection_thread.started.connect(self._detection_worker.run)

        # Po zakończeniu detekcji (sygnał `finished`):
        self._detection_worker.finished.connect(self._on_detection_finished)  # obsługa wyników
        self._detection_worker.finished.connect(self._detection_thread.quit)  # zakończenie wątku
        self._detection_worker.finished.connect(self._detection_worker.deleteLater)  # zwolnienie pamięci
        self._detection_thread.finished.connect(self._detection_thread.deleteLater)  # zwolnienie zasobów wątku

        # Startujemy wątek detekcji
        self._detection_thread.start()

    def _on_detection_finished(self, detections):
        self._last_detections = detections
        self._detection_running = False

    def _match_faces_to_trackers(self, detections):
        """
        Przypisuje nowe detekcje twarzy do istniejących trackerów (śledzonych twarzy),
        na podstawie odległości między środkami prostokątów (bounding boxów).
        Pozwala śledzić te same twarze przez wiele klatek.
        """

        # Wyciągamy prostokąty detekcji i zamieniamy na listy intów
        current_boxes = [list(map(int, det[:4])) for det in detections]
        matched = {}  # Przypisanie: face_id - index detekcji
        used_trackers = set()  # Trackery, które już zostały przypisane

        for i, box in enumerate(current_boxes):
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Środek aktualnej detekcji

            best_match = None
            best_dist = float('inf')

            # Przeszukiwanie istniejących trackerów w celu znalezienia najlepszego dopasowania
            for face_id, tracker in self.face_trackers.items():
                if face_id in used_trackers:
                    continue  # Dany tracker już został przypisany

                last_box = tracker['last_box']
                last_center = ((last_box[0] + last_box[2]) // 2, (last_box[1] + last_box[3]) // 2)
                dist = np.linalg.norm(np.array(center) - np.array(last_center))  # Odległość euklidesowa

                # Maksymalna dopuszczalna odległość zależna od czasu "zniknięcia" twarzy
                time_since_last = time.time() - tracker['last_seen']
                max_dist = 100 if time_since_last < self.face_reappear_threshold else 50

                # Wybieramy najbliższą pasującą twarz
                if dist < max_dist and dist < best_dist:
                    best_match = face_id
                    best_dist = dist

            if best_match is not None:
                # Zapisz przypisanie detekcji do istniejącego trackera
                matched[best_match] = i
                used_trackers.add(best_match)
            else:
                # Nowa twarz – przypisujemy jej nowe ID
                new_id = self.next_face_id
                while new_id in self.face_trackers:
                    new_id += 1
                self.next_face_id = new_id + 1
                matched[new_id] = i

        # Usuwanie nieaktywowanych trackerów, które wygasły (nie były widziane przez dłuższy czas)
        current_time = time.time()
        to_delete = [
            face_id for face_id, tracker in self.face_trackers.items()
            if current_time - tracker['last_seen'] > self.recognition_memory_time
        ]

        for face_id in to_delete:
            del self.face_trackers[face_id]

        return matched  # Zwracamy mapowanie: face_id - indeks detekcji

    def _handle_add_face_from_frame(self, frame: np.ndarray, name: str):
        """Obsługa dodania nowej twarzy do bazy rozpoznawania"""
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
        """
        Zwraca wyciętą twarz z podanej klatki
        """
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
        """Zapisuje log rozpoznań do pliku CSV"""
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
        """Odtwarza dźwięk ostrzeżenia jeśli wykryto nieznaną twarz"""
        now = time.time()
        if now - self.last_alert_sound_time < self.alert_cooldown:
            return

        if self.alert_effect.isPlaying():
            return

        self.logger.info("[DŹWIĘK] Odtwarzanie alertu...")
        self.alert_effect.play()
        self.last_alert_sound_time = now






