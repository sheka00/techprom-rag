import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import traceback
from .config import Config
from .rag_service import MultimodalRAG
from .image_processing import DoclingDocxChunker, PyMuPDF4LLMPDFChunker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

rag_instance: Optional[MultimodalRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_instance
    logger.info("🚀 Запуск сервера. Инициализация RAG системы в lifespan...")
    try:
        config = Config()
        rag_instance = MultimodalRAG(config)
        logger.info("✅ RAG система успешно инициализирована при запуске")
    except Exception as e:
        logger.error(f"❌ Ошибка при инициализации RAG в lifespan: {e}")

    yield

    if rag_instance:
        await rag_instance.embeddings.close()
        logger.info("👋 Завершение работы. Ресурсы очищены.")


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None
    system_prompt: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    source: str
    quote: str
    sources: list
    images_used: int
    chunks_retrieved: int
    collection_used: Optional[str] = None


@app.post("/upload", status_code=200)
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = None,
    ollama_model: Optional[str] = None,
    device: str = "cuda",
):
    global rag_instance

    logger.info("=" * 60)
    logger.info("📤 ПОЛУЧЕН ЗАПРОС НА ЗАГРУЗКУ ДОКУМЕНТА")
    logger.info("=" * 60)
    logger.info(f"📄 Файл: {file.filename}")
    logger.info(f"📋 Имя документа: {document_name or 'не указано'}")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".docx", ".pdf"]:
        logger.error(f"❌ Ошибка: файл {file.filename} не поддерживается")
        raise HTTPException(
            status_code=400, detail="Поддерживаются только DOCX и PDF файлы"
        )

    if not document_name:
        document_name = Path(file.filename).stem

    # Сохраняем файл
    logger.info("💾 Сохранение файла...")
    docx_path = f"/tmp/{file.filename}"
    with open(docx_path, "wb") as f:
        content = await file.read()
        f.write(content)
    file_size_mb = len(content) / (1024 * 1024)
    logger.info(f"   ✅ Файл сохранен: {docx_path} ({file_size_mb:.2f} MB)")

    try:
        config = Config(
            ollama_model=ollama_model or Config().ollama_model, embedding_device=device
        )

        logger.info("\n" + "🚀" * 30)
        logger.info("НАЧАЛО ОБРАБОТКИ ДОКУМЕНТА")
        logger.info("🚀" * 30 + "\n")

        logger.info(
            f"1️⃣  Конвертация {file_ext.upper()} → Markdown и сохранение изображений..."
        )
        if file_ext == ".docx":
            chunker = DoclingDocxChunker(
                docx_path=docx_path,
                document_name=document_name,
                images_dir=config.image_dir,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                debug_dir=config.debug_dir,
            )
        else:  # .pdf
            chunker = PyMuPDF4LLMPDFChunker(
                pdf_path=docx_path,
                document_name=document_name,
                images_dir="data/pdf_images",
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                debug_dir=config.debug_dir,
            )
        chunks = chunker.get_chunks()
        logger.info(f"   ✅ Создано {len(chunks)} чанков для индексации")

        logger.info("\n2️⃣  Инициализация RAG системы...")
        logger.info("   ⏳ Загрузка модели эмбеддингов...")
        if (
            rag_instance is None
            or rag_instance.config.collection_name != config.collection_name
        ):
            rag_instance = MultimodalRAG(config)
        logger.info("   ✅ RAG система инициализирована")

        logger.info("\n3️⃣  Индексация документов в Qdrant...")
        # При загрузке одного документа НЕ стираем всё остальное (recreate=False)
        stats = await rag_instance.index_documents(chunks, recreate=False)

        logger.info("\n" + "✅" * 30)
        logger.info("🎉 ДОКУМЕНТ УСПЕШНО ИНДЕКСИРОВАН!")
        logger.info("✅" * 30)
        logger.info("\n📊 Статистика:")
        logger.info(f"   • Всего чанков:    {stats['total_chunks']}")
        logger.info(f"   • Загружено:       {stats['chunks_uploaded']}")
        logger.info(f"   • С изображениями: {stats['chunks_with_images']}")
        logger.info("=" * 60)

        return {"status": "success", "document_name": document_name, "stats": stats}
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке документа: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    global rag_instance

    logger.info("=" * 60)
    logger.info("❓ ПОЛУЧЕН ЗАПРОС НА ПОИСК")
    logger.info(f"📝 Вопрос: {request.question}")
    if request.collection_name:
        logger.info(f"📋 Запрошена коллекция: {request.collection_name}")
    logger.info("=" * 60)

    # Определяем, какую коллекцию использовать
    target_collection = request.collection_name

    # Если коллекция не указана, пытаемся использовать текущий rag_instance
    if not target_collection:
        if rag_instance is not None:
            target_collection = rag_instance.config.collection_name
        else:
            # Если rag_instance нет, берем из конфига по умолчанию
            target_collection = Config().collection_name

    # Если запрошенная коллекция отличается от текущей или rag_instance еще нет
    if rag_instance is None or rag_instance.config.collection_name != target_collection:
        logger.info(f"🔄 Инициализация RAG для коллекции: {target_collection}")
        try:
            config = Config(collection_name=target_collection)
            rag_instance = MultimodalRAG(config)
            logger.info(f"✅ RAG инициализирован для {target_collection}")
        except Exception as e:
            logger.error(f"❌ Ошибка при инициализации RAG: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка инициализации коллекции {target_collection}: {str(e)}",
            )

    try:
        logger.info(f"🔍 Поиск в коллекции {target_collection}...")
        result = await rag_instance.query(
            request.question, system_prompt=request.system_prompt
        )
        result["collection_used"] = target_collection
        logger.info(f"✅ Найдено {result['chunks_retrieved']} релевантных фрагментов")
        logger.info(f"🖼️ Использовано изображений: {result['images_used']}")
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """Возвращает список всех документов в базе"""
    global rag_instance
    if rag_instance is None:
        try:
            rag_instance = MultimodalRAG(Config())
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Ошибка инициализации RAG: {e}"
            )

    docs = await rag_instance.list_indexed_documents()
    return {"documents": docs}


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """Удаляет документ из базы"""
    global rag_instance
    if rag_instance is None:
        rag_instance = MultimodalRAG(Config())

    success = await rag_instance.delete_document(doc_name)
    if success:
        return {"status": "success", "message": f"Документ {doc_name} удален"}
    else:
        raise HTTPException(
            status_code=500, detail=f"Не удалось удалить документ {doc_name}"
        )


@app.get("/health")
async def health():
    return {"status": "ok", "rag_initialized": rag_instance is not None}


class FolderUploadRequest(BaseModel):
    folder_path: str
    collection_name: str = "rag_ti_docs"
    recreate_collection: bool = True
    ollama_model: Optional[str] = None
    device: str = "cuda"
    chunk_size: int = 1500
    chunk_overlap: int = 150


@app.post("/upload_folder", status_code=200)
async def upload_folder(request: FolderUploadRequest):
    """
    Пакетная индексация: сканирует папку folder_path, находит все .docx файлы
    и индексирует их в одну коллекцию Qdrant.
    """
    global rag_instance

    folder = Path(request.folder_path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Папка не найдена: {request.folder_path}"
        )

    docx_files = sorted(folder.rglob("*.docx"))
    pdf_files = sorted(folder.rglob("*.pdf"))
    all_files = docx_files + pdf_files

    if not all_files:
        raise HTTPException(
            status_code=400,
            detail=f"Файлы DOCX или PDF не найдены в {request.folder_path}",
        )

    logger.info("=" * 60)
    logger.info("📁 ПАКЕТНАЯ ЗАГРУЗКА ДОКУМЕНТОВ")
    logger.info(f"   Папка: {request.folder_path}")
    logger.info(f"   Коллекция: {request.collection_name}")
    logger.info(
        f"   Действие: {'ПЕРЕСОЗДАНИЕ' if request.recreate_collection else 'ДОБАВЛЕНИЕ'}"
    )
    logger.info(
        f"   Найдено файлов: {len(all_files)} (DOCX: {len(docx_files)}, PDF: {len(pdf_files)})"
    )
    logger.info("=" * 60)

    # Инициализация конфигурации
    config = Config(
        ollama_model=request.ollama_model or Config.ollama_model,
        embedding_device=request.device,
        collection_name=request.collection_name,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    logger.info("🔄 Инициализация RAG системы...")
    rag_instance = MultimodalRAG(config)

    all_chunks = []
    results: List[dict] = []
    failed: List[str] = []

    for i, file_path in enumerate(all_files, 1):
        doc_name = file_path.stem
        logger.info(f"\n[{i}/{len(all_files)}] 📄 Обработка: {file_path.name}")
        try:
            if file_path.suffix.lower() == ".docx":
                chunker = DoclingDocxChunker(
                    docx_path=str(file_path),
                    document_name=doc_name,
                    images_dir="docx_images",
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                    debug_dir=config.debug_dir,
                )
            elif file_path.suffix.lower() == ".pdf":
                chunker = PyMuPDF4LLMPDFChunker(
                    pdf_path=str(file_path),
                    document_name=doc_name,
                    images_dir="data/pdf_images",
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                    debug_dir=config.debug_dir,
                )
            else:
                continue

            chunks = chunker.get_chunks()
            all_chunks.extend(chunks)
            results.append(
                {"file": file_path.name, "chunks": len(chunks), "status": "ok"}
            )
            logger.info(f"   ✅ Чанков: {len(chunks)}")
        except Exception as e:
            logger.error(f"   ❌ Ошибка: {e}")
            failed.append(file_path.name)
            results.append(
                {"file": file_path.name, "chunks": 0, "status": f"error: {e}"}
            )

    logger.info(f"\n📚 Всего чанков для индексации: {len(all_chunks)}")
    logger.info(f"⬆️  Индексация в Qdrant (recreate={request.recreate_collection})...")

    stats = await rag_instance.index_documents(
        all_chunks, recreate=request.recreate_collection
    )

    logger.info("=" * 60)
    logger.info("🎉 ПАКЕТНАЯ ЗАГРУЗКА ЗАВЕРШЕНА")
    logger.info(
        f"   Обработано файлов: {len(all_files) - len(failed)}/{len(all_files)}"
    )
    logger.info(f"   Всего чанков:      {stats['total_chunks']}")
    logger.info(f"   Загружено:         {stats['chunks_uploaded']}")
    if failed:
        logger.warning(f"   Ошибки в файлах:   {failed}")
    logger.info("=" * 60)

    return {
        "status": "success",
        "collection_name": request.collection_name,
        "total_files": len(all_files),
        "processed_files": len(all_files) - len(failed),
        "failed_files": failed,
        "stats": stats,
        "details": results,
    }


@app.get("/evaluation/info")
async def evaluation_info():
    return {
        "criterion": {
            "retrieval_accuracy": {
                "description": "Точность Retriever — процент запросов, где релевантный чанк попал в топ-5",
                "formula": "Retrieval Accuracy = (найденных) / (всего запросов)",
                "range": "0.0 - 1.0",
            }
        }
    }
