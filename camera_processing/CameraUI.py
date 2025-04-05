import cv2
import numpy as np
from typing import List, Dict, Optional


class CameraUI:
    def __init__(self, window_name="Aplikacja Kamery - Detekcja i Rozpoznawanie Twarzy"):
        self.window_name = window_name
        self.panel_width = 450  # Zwiększona szerokość dla dodatkowych informacji
        self.max_faces_display = 5
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)

        # Kolory interfejsu
        self.COLOR_BG = (240, 240, 240)  # Szare tło
        self.COLOR_TEXT = (0, 0, 0)  # Czarny tekst
        self.COLOR_MATCH = (0, 180, 0)  # Zielony dla dopasowań
        self.COLOR_UNKNOWN = (0, 0, 255)  # Czerwony dla nieznanych
        self.COLOR_HIGHLIGHT = (255, 165, 0)  # Pomarańczowy dla zaznaczenia

        # Stan interfejsu
        self.selected_face_idx = -1  # Aktualnie wybrana twarz
        self.add_face_mode = False  # Tryb dodawania nowej twarzy

    def create_right_panel(self, width: int, height: int,
                           detected_faces: List[np.ndarray],
                           recognitions: List[Dict]) -> np.ndarray:
        """
        Tworzy panel boczny z wykrytymi twarzami i opcjami dodawania

        Args:
            width: Szerokość panelu
            height: Wysokość panelu
            detected_faces: Lista wykrytych twarzy (obrazy numpy)
            recognitions: Lista słowników z informacjami o rozpoznaniu
        """
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = self.COLOR_BG

        # Nagłówek panelu
        cv2.putText(panel, "Wykryte twarze (n-dodaj, 1-5-wybierz):", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)

        # Wyświetlanie twarzy w panelu
        y_offset = 60
        face_width = int(width * 0.8)
        face_height = (height - 200) // self.max_faces_display  # Zwiększona przestrzeń

        for i, (face, rec_info) in enumerate(zip(detected_faces[:self.max_faces_display], recognitions)):
            try:
                if face.size == 0:
                    continue

                # Przygotowanie informacji o rozpoznaniu
                name = rec_info.get('name', 'Unknown')
                score = rec_info.get('score', 0)
                color = self.COLOR_MATCH if name != 'Unknown' else self.COLOR_UNKNOWN
                score_text = f"{score * 100:.1f}%" if score > 0 else "N/A"

                # Skalowanie twarzy
                h, w = face.shape[:2]
                scale = min(face_width / w, face_height / h)
                face = cv2.resize(face, None, fx=scale, fy=scale)
                face_h, face_w = face.shape[:2]

                # Zaznaczenie wybranej twarzy
                border_color = self.COLOR_HIGHLIGHT if i == self.selected_face_idx else color
                border_thickness = 3 if i == self.selected_face_idx else 2

                # Obramowanie i twarz
                cv2.rectangle(panel, (10, y_offset - 5),
                              (10 + face_w + 10, y_offset + face_h + 5),
                              border_color, border_thickness)
                panel[y_offset:y_offset + face_h, 15:15 + face_w] = face

                # Informacje o rozpoznaniu
                text_y = y_offset + face_h + 25
                cv2.putText(panel, f"{i + 1}. {name}", (15, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(panel, f"Podobieństwo: {score_text}", (15, text_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                y_offset += face_h + 70

            except Exception as e:
                print(f"Błąd wyświetlania twarzy {i}: {str(e)}")
                continue

        # Instrukcja dodawania nowej twarzy
        if self.add_face_mode and self.selected_face_idx >= 0:
            cv2.putText(panel, "Wprowadź imię i naciśnij ENTER:",
                        (15, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        self.COLOR_HIGHLIGHT, 1)
            cv2.putText(panel, "(ESC aby anulować)",
                        (15, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (100, 100, 100), 1)

        return panel

    def display_frame(self, frame: np.ndarray,
                      detected_faces: List[np.ndarray] = [],
                      recognitions: Optional[List[Dict]] = None):
        """Wyświetla klatkę z panelem bocznym"""
        recognitions = recognitions or [{} for _ in detected_faces]
        right_panel = self.create_right_panel(self.panel_width, frame.shape[0],
                                              detected_faces, recognitions)
        combined = np.hstack((frame, right_panel))
        cv2.imshow(self.window_name, combined)

    def handle_key_events(self) -> Optional[Dict]:
        """
        Obsługuje zdarzenia klawiatury i zwraca akcje do wykonania

        Returns:
            Dict: {'action': 'add_face', 'face_idx': int, 'name': str}
                  lub None jeśli brak akcji
        """
        key = cv2.waitKey(1) & 0xFF

        # Wybór twarzy klawiszami 1-5
        if ord('1') <= key <= ord(str(min(5, self.max_faces_display))):
            self.selected_face_idx = key - ord('1')
            self.add_face_mode = False
            return None

        # Wejście w tryb dodawania twarzy
        if key == ord('n') and self.selected_face_idx >= 0:
            self.add_face_mode = True
            return None

        # Anulowanie trybu dodawania
        if key == 27:  # ESC
            self.add_face_mode = False
            return None

        # Potwierdzenie dodania twarzy
        if self.add_face_mode and key == 13:  # ENTER
            # Otwórz okno dialogowe do wprowadzenia imienia
            name = self._get_face_name_from_input()
            if name:
                action = {
                    'action': 'add_face',
                    'face_idx': self.selected_face_idx,
                    'name': name
                }
                self.add_face_mode = False
                return action

        return None

    def _get_face_name_from_input(self) -> Optional[str]:
        """Wyświetla okno dialogowe do wprowadzenia imienia"""
        name = input("Wprowadź imię osoby do dodania (bez polskich znaków i spacji): ")
        return name.strip() if name else None

    def should_close(self) -> bool:
        """Sprawdza czy aplikacja powinna się zamknąć"""
        return (cv2.waitKey(1) == ord('q') or
                cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1)