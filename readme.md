
# System Rozpoznawania Twarzy z Obsługą Zakłóceń (Okulary)

Projekt desktopowej aplikacji do rozpoznawania twarzy w czasie rzeczywistym i na nagraniach wideo. System został zbudowany w Pythonie z wykorzystaniem bibliotek: PyQt6, OpenCV, PyTorch oraz FaceX-Zoo.

## Struktura katalogów

```

rozpoznawanie-twarzy/
├── api\_usage/              # Przykłady użycia API FaceX-Zoo
├── camera\_processing/      # Przetwarzanie obrazu z kamery
├── config/                 # Pliki konfiguracyjne (.yaml)
├── core/                   # Klasy modeli i obsługa embeddingów
├── logs/                   # Pliki logów systemu
├── models/                 # Modele sieci neuronowych
├── my\_faces/               # Zarejestrowane wzorce twarzy
├── recordings/             # Przykładowe nagrania
├── sounds/                 # Dźwięki systemowe
├── utils/                  # Funkcje pomocnicze
├── main.py                 # Punkt wejścia aplikacji
├── Dockerfile              # Plik obrazu Docker
├── requirements.txt        # Zależności Pythona
```
---

## Uruchamianie aplikacji

### Wymagania

- Docker (20.10+)
- Python 3.9+ (jeśli nie używasz Dockera)
- VcXsrv (Windows) lub X11 (Linux)

---

### Windows 10/11 (z VcXsrv)

1. **Zainstaluj i uruchom VcXsrv**:
   - Użyj `XLaunch`
   - `Display Number`: `0`
   - `Start no client`
   - Zaznacz: `Disable access control`

2. **Zbuduj obraz Docker**:
   Otwórz PowerShell w katalogu projektu:

   ```bash
   docker build -t rozpoznawanie-twarzy .
    ```

3. **Uruchom aplikację**:

   ```bash
   docker run -it --rm -e DISPLAY=host.docker.internal:0.0 rozpoznawanie-twarzy
   ```

---

### Linux (Ubuntu/Debian)

1. **Zezwól Dockerowi na połączenie z X11**:

   ```bash
   xhost +local:docker
   ```

2. **Zbuduj obraz**:

   ```bash
   docker build -t rozpoznawanie-twarzy .
   ```

3. **Uruchom aplikację**:

   ```bash
   docker run -it --rm \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     rozpoznawanie-twarzy
   ```

---

## Alternatywnie: uruchomienie bez Dockera (lokalnie)

1. **Zainstaluj zależności**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Uruchom aplikację**:

   ```bash
   python main.py
   ```

---

## Przykłady danych testowych

* Pliki wideo do testów znajdują się w folderze `recordings/`
* Przykładowe wzorce twarzy w `my_faces/`

---

## Autorzy

* Patrycja Smits
* Julia Krok