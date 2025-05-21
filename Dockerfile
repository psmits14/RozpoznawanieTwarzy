FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Systemowe zależności potrzebne dla PyQt6 i opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxtst6 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglib2.0-0 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    libdbus-1-3 \
    libpulse0 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-util1 \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*



WORKDIR /app

# Skopiuj wymagania
COPY requirements.txt .

# Zainstaluj pip i torch osobno z właściwego źródła
RUN pip install --upgrade pip && \
    pip install torch==2.2.2+cpu torchvision==0.17.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Usuń cały katalog qt + qt.conf z cv2, żeby Qt używało PyQt6-owego pluginu
RUN rm -rf /usr/local/lib/python3.11/site-packages/cv2/qt && \
    rm -f /usr/local/lib/python3.11/site-packages/cv2/qt.conf

CMD ["/bin/sh", "-c", "QT_QPA_PLATFORM_PLUGIN_PATH=/usr/local/lib/python3.11/site-packages/PyQt6/Qt6/plugins/platforms QT_PLUGIN_PATH= QT_DEBUG_PLUGINS=1 python main.py"]
