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

Репозиторий настроен так, что все тяжелые веса моделей и базы документов исключены из Git (`.gitignore`). Это позволяет клонировать и разворачивать проект моментально на любом сервере.

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/sheka00/techprom-rag.git
   cd techprom-rag
   ```

2. **Запустите сборку и экспорт в фоне:**
   Мастер-скрипт автоматически проверит наличие TensorRT-модели (и скомпилирует её при необходимости, ~5-10 мин), поднимет всю Docker-инфраструктуру и скачает нужные LLM. Чтобы процесс не прервался при закрытии терминала, используйте `nohup`:
   ```bash
   chmod +x run.sh scripts/convert_trt.sh
   cp -n .env.example .env  # Создать .env, если его нет
   nohup ./run.sh > build.log 2>&1 &
   ```

3. **Следите за логами и статусом:**
   ```bash
   tail -f build.log
   docker compose ps
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

Для индексации всех документов (ТИ и iteration_3_docs) выполните:
```bash
docker exec -it techprom-rag-api-1 python3 -m app.full_reindex
```

## 📡 Ключевые возможности

### 1. Высокопроизводительные эмбеддинги
Использование **TensorRT** через Triton Server позволяет обрабатывать сотни запросов в секунду с минимальной задержкой. Модель `bge-m3` оптимизирована для юридических и технических текстов.

### 2. Мультимодальный RAG
Сервис извлекает не только текст, но и изображения из документов. При поиске наиболее релевантные изображения могут передаваться в LLM для анализа контекста.

### 3. Глобальная переиндексация
Для индексации больших объемов документов используйте встроенный скрипт:
```bash
docker exec -it techprom-rag-api-1 python3 -m app.full_reindex
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