"""Экспорт модели в ONNX. TensorRT — run.sh через convert_trt.sh."""

import os
import sys
from export_onnx import export_explicit_model

# Добавляем директорию scripts в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Пути относительно корня проекта
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models/bge_model/1/model.plan")
ONNX_PATH = os.path.join(SCRIPT_DIR, "model.onnx")


def main():
    if os.path.exists(TRT_MODEL_PATH):
        print(f"TRT модель уже есть: {TRT_MODEL_PATH}. Экспорт пропущен.")
        return

    export_explicit_model(ONNX_PATH)
    print(f"ONNX сохранён: {ONNX_PATH}. Дальше run.sh вызовет convert_trt.sh.")


if __name__ == "__main__":
    main()
