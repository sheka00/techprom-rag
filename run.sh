#!/usr/bin/env bash
# Экспорт модели (если нет) → docker compose (Qdrant + Triton + Encoder + Ollama + API + UI). Запуск: ./run.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRT_MODEL="models/bge_model/1/model.plan"
VENV_DIR="triton_export_env"

# 1. Экспорт модели в TensorRT (если ещё нет)
if [ ! -f "$TRT_MODEL" ]; then
  echo "=== TRT модели нет, выполняем экспорт ==="
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  # Нам нужны torch, transformers, onnx для экспорта
  pip install -q torch transformers onnx onnxscript
  python scripts/export_model.py
  ./scripts/convert_trt.sh model.onnx "$TRT_MODEL"
  rm -f model.onnx model.onnx.data
  deactivate
  echo "=== Удаление временного виртуального окружения ==="
  rm -rf "$VENV_DIR"
else
  echo "=== TRT модель уже есть: $TRT_MODEL ==="
fi

# 2. Остановить существующие контейнеры (если есть)
echo "=== Останавливаем существующие контейнеры ==="
docker compose down 2>/dev/null || true

# 3. Запуск всех сервисов через Docker Compose
echo "=== Запуск всех сервисов: Qdrant, Triton, Encoder, Ollama, API, UI ==="
docker compose up -d --build

# 4. Загрузка LLM моделей в Ollama (если ещё нет)
echo "=== Проверка/загрузка основной LLM qwen3:14b в Ollama ==="
docker compose exec ollama ollama pull qwen3:14b

echo "=== Проверка/загрузка модели суммаризации qwen3.5:4b в Ollama ==="
docker compose exec ollama ollama pull qwen3.5:4b

echo ""
echo "=== Сервисы запущены ==="
echo "Triton Server:"
echo "  - HTTP: http://localhost:8000"
echo ""
echo "Encoder Service (FastAPI wrapper):"
echo "  - API: http://localhost:8080"
echo "  - Health: curl localhost:8080/health"
echo ""
echo "Main API:"
echo "  - URL: http://localhost:8005"
echo ""
echo "Gradio UI:"
echo "  - URL: http://localhost:7860"
echo ""
echo "Проверка статуса: docker compose ps"
