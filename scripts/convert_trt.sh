
#!/usr/bin/env bash
# Конвертация ONNX в TensorRT через trtexec в Docker.
# Использование: ./convert_trt.sh [путь к ONNX] [путь к model.plan]
# По умолчанию: model.onnx -> models/bge_model/1/model.plan

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ONNX_PATH="${1:-$PROJECT_ROOT/model.onnx}"
TRT_OUTPUT="${2:-$PROJECT_ROOT/models/bge_model/1/model.plan}"

mkdir -p "$(dirname "$TRT_OUTPUT")"

docker run --gpus all --rm \
  -v "$PROJECT_ROOT:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/tensorrt:25.01-py3 \
  trtexec \
  "--onnx=$ONNX_PATH" \
  "--saveEngine=$TRT_OUTPUT" \
  --memPoolSize=workspace:4096 \
  --fp16
