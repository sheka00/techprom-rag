# 🚀 Мультимодальный RAG + BGE-M3 на Triton Inference Server

Комплексная система для поиска и извлечения информации из документов (DOCX, PDF) с поддержкой изображений и высокопроизводительным инференсом на TensorRT.

## 📋 Описание проекта

Проект включает в себя:
- **RAG-сервис (FastAPI)** — основная логика поиска, парсинг документов (Docling/PyMuPDF4LLM), индексация в Qdrant и генерация ответов через Ollama.
- **Triton Inference Server** — инференс модели эмбеддингов `bge-m3` в формате TensorRT для максимальной производительности на GPU.
- **Encoder Service (FastAPI wrapper)** — легковесный микросервис-обёртка для токенизации и связи с Triton, что изолирует основное приложение от тяжелых зависимостей (torch/transformers).
- **Автоматизированный экспорт** — скрипты для перевода HuggingFace модели в ONNX → TensorRT.

## 📁 Структура проекта

```bash
.
├── run.sh                  # ЕДИНАЯ ТОЧКА ЗАПУСКА: экспорт + поднятие всех контейнеров
├── docker-compose.yml      # Оркестрация: Qdrant, Triton, Encoder, Ollama, API, UI
├── app/                    # Основной RAG-сервис (FastAPI)
├── service/                # Микросервис энкодера (FastAPI + Triton client)
├── scripts/                # Скрипты экспорта модели в ONNX/TensorRT
├── models/                 # Репозиторий моделей Triton (TensorRT .plan файлы)
├── client/                 # Python-клиент для прямого вызова Encoder Service
├── data/                   # Данные (документы, изображения, БД)
└── gradio_app.py           # Веб-интерфейс для чата с документами
```

## 🚀 Быстрый старт

Просто запустите мастер-скрипт. Он автоматически проверит наличие TensorRT-модели (и экспортирует её при необходимости, ~5-10 мин), поднимет всю инфраструктуру и **скачает LLM модель qwen3:8b в Ollama**.

```bash
chmod +x run.sh scripts/convert_trt.sh
cp -n .env.example .env  # Создать .env, если его нет
./run.sh
```

**Сервисы будут доступны по адресам:**
- **RAG API:** [http://localhost:8005](http://localhost:8005)
- **Gradio UI:** [http://localhost:7860](http://localhost:7860)
- **Triton Server (metrics):** [http://localhost:8002/metrics](http://localhost:8002/metrics)
- **Encoder (health):** [http://localhost:8080/health](http://localhost:8080/health)

## 📂 Подготовка данных

Поместите ваши документы (DOCX, PDF) в папку `data/`. Для работы системы понадобятся следующие датасеты (они добавлены в `.gitignore` из-за большого объема):
- **`data/ТИ/`** — папка с Техническими Инструкциями.
- **`data/iteration_3_docs/`** — расширенная база документов для третьего этапа.

Для индексации документов в Qdrant выполните:
```bash
docker exec -it techprom_rag-main-api-1 python3 -m app.reindex \
  --folder data/название_вашей_папки --collection rag_ti_docs
```

## 📡 Ключевые возможности

### 1. Высокопроизводительные эмбеддинги
Использование **TensorRT** через Triton Server позволяет обрабатывать сотни запросов в секунду с минимальной задержкой. Модель `bge-m3` оптимизирована для юридических и технических текстов.

### 2. Мультимодальный RAG
Сервис извлекает не только текст, но и изображения из документов. При поиске наиболее релевантные изображения могут передаваться в LLM для анализа контекста.

### 3. Глобальная переиндексация
Для индексации больших объемов документов используйте встроенный скрипт:
```bash
docker exec -it techprom_rag-main-api-1 python3 -m app.reindex \
  --folder data/docs --collection my_collection
```

### 4. Бенчмаркинг
Оцените качество поиска и ответов с помощью `LLM-судьи`:
```bash
docker compose exec api python -m app.run_benchmark --docx data/test_questions.docx
```

## 🛠 Разработка и тестирование

**Ручная проверка эмбеддингов (через Encoder Service):**
```bash
curl -X POST http://localhost:8080/encode -H "Content-Type: application/json" -d '{"query": "привет"}'
```

**Нагрузочное тестирование (Locust):**
```bash
pip install locust
locust -f locustfile.py --host http://localhost:8080 --headless -u 50 -r 5 -t 60s
```

---

## ⚙️ Требования
- **ОС:** Linux (рекомендуется)
- **GPU:** NVIDIA (с NVIDIA Container Toolkit)
- **ПО:** Docker Compose v2, Python 3.12+ (для локального экспорта моделей)