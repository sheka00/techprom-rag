import os
import logging
import asyncio
from pathlib import Path
from qdrant_client import QdrantClient
from app.config import Config
from app.rag_service import MultimodalRAG
from app.image_processing import DoclingDocxChunker, PyMuPDF4LLMPDFChunker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FullReindexing")


async def run_full_reindexing():
    folders = ["data/ТИ", "data/iteration_3_docs"]
    collection_name = "rag_ti_docs"

    # Резолвим URL Qdrant (берем из окружения или дефолт)
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    print(f"🚀 Инициализация... Qdrant: {qdrant_url}", flush=True)

    # 1. Удаление ВСЕХ коллекций
    try:
        client = QdrantClient(url=qdrant_url, timeout=10.0)
        collections = client.get_collections().collections
        print(f"🗑️ Удаление существующих коллекций ({len(collections)})...", flush=True)
        for c in collections:
            print(f"   Удаляю: {c.name}", flush=True)
            client.delete_collection(c.name)
        print("✅ Все коллекции удалены.", flush=True)
    except Exception as e:
        print(f"❌ Ошибка при очистке коллекций: {e}", flush=True)
        return

    # 2. Сбор файлов из всех папок (проверка на дубликаты по имени)
    all_files = []
    seen_names = set()

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"⚠️ Папка не найдена: {folder}")
            continue

        logger.info(f"📂 Сканирование папки: {folder}")
        docx = list(folder_path.rglob("*.docx"))
        pdf = list(folder_path.rglob("*.pdf"))

        for f in docx + pdf:
            if f.name in seen_names:
                # В iteration_3_docs могут быть обновленные версии тех же файлов,
                # поэтому мы могли бы захотеть их заменить.
                # Но пока просто добавим всё, MultimodalRAG использует stem как doc_name.
                # Если имена совпадают, это может привести к дублям в 'document_name'.
                # Однако пользователь хочет "всё из двух папок".
                logger.info(
                    f"   ℹ️ Файл {f.name} уже встречался, добавляю копию из {folder}"
                )

            all_files.append(f)
            seen_names.add(f.name)

    if not all_files:
        logger.error("❌ Файлы не найдены.")
        return

    logger.info(f"📚 Найдено всего файлов: {len(all_files)}")

    # 3. Инициализация RAG и индексация
    config = Config(
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        embedding_device="cuda",  # Пытаемся использовать CUDA
    )

    rag_instance = MultimodalRAG(config)

    all_chunks = []
    for i, file_path in enumerate(all_files, 1):
        doc_name = file_path.stem
        logger.info(f"[{i}/{len(all_files)}] 📄 Обработка: {file_path.name}")
        try:
            if file_path.suffix.lower() == ".docx":
                chunker = DoclingDocxChunker(
                    docx_path=str(file_path),
                    document_name=doc_name,
                    images_dir="docx_images",
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
            elif file_path.suffix.lower() == ".pdf":
                chunker = PyMuPDF4LLMPDFChunker(
                    pdf_path=str(file_path),
                    document_name=doc_name,
                    images_dir="data/pdf_images",
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
            else:
                continue

            chunks = chunker.get_chunks()
            all_chunks.extend(chunks)
            logger.info(f"   ✅ Чанков: {len(chunks)}")
        except Exception as e:
            logger.error(f"   ❌ Ошибка при обработке {file_path.name}: {e}")

    logger.info(f"\n🚀 Загрузка {len(all_chunks)} чанков в Qdrant...")
    stats = await rag_instance.index_documents(all_chunks, recreate=True)

    logger.info("=" * 60)
    logger.info("🎉 ГЛОБАЛЬНАЯ ПЕРЕИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    logger.info(f"   Всего чанков:      {stats['total_chunks']}")
    logger.info(f"   Загружено:         {stats['chunks_uploaded']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_full_reindexing())
