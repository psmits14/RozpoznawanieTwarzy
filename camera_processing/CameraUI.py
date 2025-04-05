import cv2
import numpy as np


class CameraUI:
    def __init__(self, window_name="Aplikacja Kamery z Detekcją i Rozpoznawaniem Twarzy"):
        self.window_name = window_name
        self.panel_width = 400  # Zwiększona szerokość dla dodatkowych informacji
        self.max_faces_display = 5
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)

        # Kolory dla różnych elementów interfejsu
        self.COLOR_BG = (240, 240, 240)  # Szare tło
        self.COLOR_TEXT = (0, 0, 0)  # Czarny tekst
        self.COLOR_MATCH = (0, 180, 0)  # Zielony dla dopasowań
        self.COLOR_UNKNOWN = (0, 0, 255)  # Czerwony dla nieznanych

    def create_right_panel(self, width, height, detected_faces, recognitions=None):
        """
        Tworzy panel boczny z wykrytymi twarzami i informacjami o rozpoznaniu

        Args:
            width: Szerokość panelu
            height: Wysokość panelu
            detected_faces: Lista wykrytych twarzy (obrazy numpy)
            recognitions: Lista słowników z informacjami o rozpoznaniu
                         [{'name': str, 'score': float}, ...]
        """
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = self.COLOR_BG

        # Nagłówek panelu
        cv2.putText(panel, "Wykryte i rozpoznane twarze:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)

        # Wyświetlanie twarzy w panelu
        y_offset = 60
        max_face_width = int(width * 0.8)  # Mniejsza szerokość dla dwóch kolumn
        max_face_height = (height - 150) // self.max_faces_display

        recognitions = recognitions or [{} for _ in detected_faces]  # Domyślne puste rozpoznania

        for i, (face, rec_info) in enumerate(zip(detected_faces[:self.max_faces_display], recognitions)):
            try:
                if face.size == 0:
                    continue

                # Przygotuj informacje o rozpoznaniu
                name = rec_info.get('name', 'Nieznany')
                score = rec_info.get('score', 0)
                color = self.COLOR_MATCH if name != 'Nieznany' else self.COLOR_UNKNOWN
                score_text = f"{score * 100:.1f}%" if score > 0 else "N/A"

                # Skalowanie zachowujące proporcje
                h, w = face.shape[:2]
                scale = min(max_face_width / w, max_face_height / h)
                face = cv2.resize(face, None, fx=scale, fy=scale)
                face_height, face_width = face.shape[:2]

                # Pozycja dla obrazu i tekstu
                x_pos = 20  # Stałe wcięcie od lewej

                # Rysowanie obramowania wokół twarzy
                border_color = color if name != 'Nieznany' else (100, 100, 100)
                cv2.rectangle(panel,
                              (x_pos - 2, y_offset - 2),
                              (x_pos + face_width + 2, y_offset + face_height + 2),
                              border_color, 2)

                # Umieszczenie twarzy w panelu
                panel[y_offset:y_offset + face_height, x_pos:x_pos + face_width] = face

                # Informacje o rozpoznaniu
                text_y = y_offset + face_height + 30
                cv2.putText(panel, f"Osoba: {name}", (x_pos, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(panel, f"Podobieństwo: {score_text}", (x_pos, text_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                y_offset += face_height + 70  # Więcej miejsca na informacje

            except Exception as e:
                print(f"Błąd przetwarzania twarzy {i}: {str(e)}")
                continue

        return panel

    def display_frame(self, frame, detected_faces=[], recognitions=None):
        """Wyświetla klatkę z panelem bocznym"""
        right_panel = self.create_right_panel(self.panel_width, frame.shape[0], detected_faces, recognitions)
        combined = np.hstack((frame, right_panel))
        cv2.imshow(self.window_name, combined)

    def should_close(self):
        """Sprawdza czy aplikacja powinna się zamknąć"""
        return (cv2.waitKey(1) == ord('q') or
                cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1)