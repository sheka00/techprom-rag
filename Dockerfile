FROM python:3.12-slim

WORKDIR /app

# Системные зависимости для docling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    pandoc \
    libreoffice-writer-nogui \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Установка зависимостей из requirements.txt
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

# Принудительная установка Torch с поддержкой CUDA 12.8 (если требуется для RTX 5060 Ti)
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128


COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
