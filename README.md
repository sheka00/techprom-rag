# 🚀 Мультимодальный RAG + BGE-M3 на Triton Inference Server

Комплексная система для поиска и извлечения информации из документов (DOCX, PDF) с поддержкой OCR скан-PDF, табличных данных и высокопроизводительным инференсом на TensorRT.

## 📋 Описание проекта

- **RAG-сервис (FastAPI)** — основная логика поиска, парсинг документов (Docling/PyMuPDF4LLM + GLM OCR fallback), индексация в Qdrant и генерация ответов через Ollama.
- **Triton Inference Server** — инференс модели эмбеддингов `bge-m3` в формате TensorRT для максимальной производительности на GPU.
- **Encoder Service (FastAPI wrapper)** — микросервис-обёртка для токенизации и связи с Triton.
- **Ollama** — хостит три модели: основную LLM, модель суммаризации и модель GLM OCR.
- **Qdrant** — векторная база данных (3 коллекции: основная, таблицы, суммаризации).

## 📁 Структура проекта

```bash
.
├── run.sh                  # ЕДИНАЯ ТОЧКА ЗАПУСКА
├── docker-compose.yml      # Оркестрация: Qdrant, Triton, Encoder, Ollama, API, UI
├── app/                    # Основной RAG-сервис (FastAPI)
│   ├── full_reindex.py     # Скрипт полной переиндексации
│   ├── rag_service.py      # Логика RAG, суммаризации, поиска
│   └── image_processing.py # Парсинг DOCX/PDF + GLM OCR fallback
├── service/                # Микросервис энкодера (FastAPI + Triton client)
├── scripts/                # Скрипты экспорта модели в ONNX/TensorRT
├── models/                 # Репозиторий моделей Triton (.plan файлы)
├── data/                   # Документы (в .gitignore, добавляются вручную)
│   ├── ТИ/                 # Технические Инструкции
│   └── iteration_3_docs/   # Расширенная база документов
└── gradio_app.py           # Веб-интерфейс (Gradio UI)
```

## 🚀 Быстрый старт (деплой с нуля)

Все тяжелые веса моделей и данные исключены из Git (`.gitignore`). Клонирование — мгновенное.

### Шаг 1: Клонировать репозиторий

```bash
git clone https://github.com/sheka00/techprom-rag.git
cd techprom-rag
```

> ⚠️ Если клонирование выполнялось от `root`, а работаете как обычный пользователь, выполните:
> ```bash
> git config --global --add safe.directory $(pwd)
> ```

### Шаг 2: Подготовить окружение

```bash
chmod +x run.sh scripts/convert_trt.sh
cp -n .env.example .env   # Создать .env из шаблона (если нет)
```

### Шаг 3: Запустить в фоне

Скрипт автоматически:
- Скомпилирует TensorRT-модель (`bge-m3`) при первом запуске (~5–10 мин)
- Поднимет все сервисы (Qdrant, Triton, Encoder, Ollama, API, UI)
- Скачает три модели в Ollama (см. ниже)

```bash
nohup ./run.sh > build.log 2>&1 &
tail -f build.log
```

### Шаг 4: Добавить документы

Скопируйте ваши DOCX/PDF в папки данных:

```bash
cp -r /path/to/your/ТИ/ data/ТИ/
cp -r /path/to/your/docs/ data/iteration_3_docs/
```

### Шаг 5: Запустить индексацию

```bash
docker exec -it techprom-rag-api-1 python3 -m app.full_reindex
```

Индексация создаёт три коллекции в Qdrant:
- `rag_ti_docs` — основные текстовые чанки
- `rag_ti_tables` — табличные данные
- `rag_ti_summaries` — суммаризированные двойные чанки (для семантического поиска)

## 🤖 Модели Ollama

`run.sh` автоматически скачивает три модели:

| Модель | Назначение | Размер |
|---|---|---|
| `qwen3:14b` | Основная LLM (генерация ответов) | ~9 GB |
| `qwen3.5:4b` | Суммаризация чанков при индексации | ~2.5 GB |
| `glm-ocr:latest` | OCR для скан-PDF (fallback) | ~8 GB |

Проверить загруженные модели:
```bash
docker exec -it techprom-rag-ollama-1 ollama list
```

## 📄 Обработка PDF

Система использует двухуровневый fallback:

1. **Docling** (основной путь) — структурированный парсинг PDF с сохранением таблиц и формул.
2. **GLM OCR** (fallback) — если Docling вернул "мусор" (скан без текстового слоя), автоматически запускается layout-детекция (`PP-DocLayoutV3`) + распознавание текста через `glm-ocr:latest` в Ollama.

## 📡 Адреса сервисов

| Сервис | Адрес |
|---|---|
| Gradio UI | http://localhost:7860 |
| RAG API | http://localhost:8005 |
| Triton metrics | http://localhost:8002/metrics |
| Encoder health | http://localhost:8080/health |
| Qdrant UI | http://localhost:6333/dashboard |

## 🔄 Переиндексация (обновление базы)

```bash
# Полная переиндексация (удаляет все коллекции и создаёт заново)
docker exec -it techprom-rag-api-1 python3 -m app.full_reindex
```

## 🛠 Разработка и диагностика

```bash
# Проверить статус контейнеров
docker compose ps

# Логи API
docker compose logs -f api

# Логи Ollama
docker compose logs -f ollama

# Проверить эмбеддинги
curl -X POST http://localhost:8080/encode \
  -H "Content-Type: application/json" \
  -d '{"query": "тест"}'

### 4. Бенчмаркинг и оценка качества

Оцените качество поиска и ответов с помощью автоматизированного теста с `LLM-судьёй`. Скрипт проходит по вопросам из DOCX-файла, запрашивает ответ у RAG и просит основную модель (qwen3:14b) оценить правильность ответа по сравнению с эталоном.

**Примеры запуска:**
```bash
# Базовый набор (157 вопросов)
docker exec -it techprom-rag-api-1 python3 -m app.run_benchmark --docx data/тестовые_вопросы.docx

# Набор Iteration 3 (76 вопросов)
docker exec -it techprom-rag-api-1 python3 -m app.run_benchmark --docx data/benchmark_iteration_3.docx --report data/benchmark_report_iteration_3.json
```

**Дополнительные параметры:**
- `--limit 10` — проверить только первые 10 вопросов (для быстрой отладки).
- `--report data/my_report.json` — сохранить результаты в другой файл.
- `--detailed` — использовать усиленный промпт для получения развернутых ответов.

**Результаты:**
После завершения в папке `data/` появится файл `benchmark_report.json` с детальной статистикой по каждому вопросу (Retrieval Hit, LLM Judge Score, цитаты).

## ⚙️ Требования

- **ОС:** Linux
- **GPU:** NVIDIA (16+ GB VRAM рекомендуется; минимум 12 GB)
- **ПО:** Docker Compose v2, NVIDIA Container Toolkit
- **Дисковое пространство:** ~30 GB (модели Ollama + TensorRT + данные)