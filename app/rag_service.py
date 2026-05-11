import base64
import os
import threading
import json
import logging
import collections
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .config import Config
from .document_parser import DocumentChunk
from .triton_embeddings import TritonEmbeddings
from qdrant_client.http import models as rest_models
import time
import torch
import traceback
import requests
import anyio
import asyncio


logger = logging.getLogger(__name__)


# Классы для эмбеддингов


class RAGResponse(BaseModel):
    """Структурированный ответ от RAG системы"""

    answer: str = Field(
        description="Подробный ответ на вопрос на основе предоставленного контекста"
    )
    source: str = Field(
        description="Название документа и конкретный раздел/пункт, где найден ответ"
    )
    quote: str = Field(
        description="Краткая точная цитата или описание фрагмента изображения, подтверждающее ответ"
    )


class MultimodalRAG:
    def __init__(self, config: Config):
        self.config = config

        # Автоматическое определение устройства с fallback на CPU
        if config.embedding_device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA запрошена, но недоступна. Используется CPU.")
                self.actual_device = "cpu"
            else:
                self.actual_device = config.embedding_device
                if ":" in config.embedding_device:
                    gpu_idx = int(config.embedding_device.split(":")[-1])
                else:
                    gpu_idx = 0
                if gpu_idx < torch.cuda.device_count():
                    logger.info(
                        f"✅ CUDA доступна: {torch.cuda.get_device_name(gpu_idx)}"
                    )
                    logger.info(
                        f"   Память GPU: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3:.2f} GB"
                    )
                else:
                    logger.warning(f"⚠️ GPU {gpu_idx} недоступна. Используется GPU 0.")
                    self.actual_device = "cuda:0"
        else:
            self.actual_device = config.embedding_device

        logger.info("🔄 Инициализация системы эмбеддингов (через Encoder Service)...")
        logger.info(f"   URL: {config.encoder_url}")

        try:
            start_time = time.time()
            # Всегда используем TritonEmbeddings через внешний сервис токенизации
            self.embeddings = TritonEmbeddings(
                model_name=config.embedding_model,
                triton_url=config.triton_url,
                triton_model=config.triton_model,
                encoder_url=config.encoder_url,
                batch_size=64 if self.actual_device.startswith("cuda") else 16,
            )

            elapsed = time.time() - start_time
            logger.info(f"   ✅ Система эмбеддингов готова за {elapsed:.1f} секунд")
        except Exception as e:
            logger.error(f"   ❌ Ошибка при инициализации эмбеддингов: {e}")
            logger.error(traceback.format_exc())
            raise

        if config.qdrant_url:
            self.qdrant_client = QdrantClient(url=config.qdrant_url, timeout=60.0)
            self.async_qdrant_client = AsyncQdrantClient(
                url=config.qdrant_url, timeout=60.0
            )
        elif config.qdrant_path:
            self.qdrant_client = QdrantClient(path=config.qdrant_path, timeout=60.0)
            self.async_qdrant_client = AsyncQdrantClient(
                path=config.qdrant_path, timeout=60.0
            )
        else:
            self.qdrant_client = QdrantClient(":memory:", timeout=60.0)
            self.async_qdrant_client = AsyncQdrantClient(":memory:", timeout=60.0)

        # Инициализируем vector_store и retriever из существующей коллекции, если она есть
        try:
            collections_info = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections_info]

            # Основная коллекция
            if config.collection_name in collection_names:
                logger.info(
                    f"✅ Найдена существующая коллекция {config.collection_name}, инициализирую retriever..."
                )
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=config.collection_name,
                    embedding=self.embeddings,
                )
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": config.retriever_k}
                )
                logger.info("✅ Retriever инициализирован из существующей коллекции")
            else:
                self.vector_store = None
                self.retriever = None

            # Табличная коллекция
            if config.tables_collection_name in collection_names:
                logger.info(
                    f"✅ Найдена табличная коллекция {config.tables_collection_name}, инициализирую retriever..."
                )
                self.tables_vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=config.tables_collection_name,
                    embedding=self.embeddings,
                )
                self.tables_retriever = self.tables_vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": config.tables_retriever_k},
                )
                logger.info("✅ Табличный retriever инициализирован")
            else:
                self.tables_vector_store = None
                self.tables_retriever = None

            # Коллекция суммаризаций
            if config.summaries_collection_name in collection_names:
                logger.info(
                    f"✅ Найдена коллекция суммаризаций {config.summaries_collection_name}, инициализирую retriever..."
                )
                self.summaries_vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=config.summaries_collection_name,
                    embedding=self.embeddings,
                )
                self.summaries_retriever = self.summaries_vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": config.summary_retriever_k},
                )
                logger.info("✅ Retriever суммаризаций инициализирован")
            else:
                self.summaries_vector_store = None
                self.summaries_retriever = None

        except Exception as e:
            logger.warning(f"⚠️ Не удалось инициализировать retriever: {e}")
            self.vector_store = None
            self.retriever = None
            self.tables_vector_store = None
            self.tables_retriever = None

        self.image_cache = {}

        # ИНИЦИАЛИЗАЦИЯ МОДЕЛИ OLLAMA
        self.llm = ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=0.0,
            timeout=config.ollama_timeout,
            num_ctx=config.ollama_num_ctx,  # Размер контекста из конфига
            seed=config.ollama_seed,
            extra_kwargs={
                "format": "json"
            },  # Оставляем для надежности на уровне Ollama
        )
        # Привязываем схему ответа
        self.structured_llm = self.llm.with_structured_output(RAGResponse)

        # Предпрогрев модели для мгновенного ответа
        try:
            logger.info(f"   🔥 Предпрогрев модели {config.ollama_model}...")

            # Отправляем пустой запрос для загрузки модели в память
            def warm_up():
                try:
                    # Небольшая пауза перед запросом, чтобы Ollama успела инициализироваться
                    time.sleep(5)

                    response = requests.post(
                        f"{config.ollama_base_url}/api/generate",
                        json={
                            "model": config.ollama_model,
                            "prompt": "",
                            "keep_alive": -1,  # Гарантируем бесконечное нахождение в памяти
                        },
                        timeout=60,  # Даем больше времени на загрузку с диска в VRAM
                    )

                    if response.status_code == 200:
                        # Проверяем, в какой памяти находится модель (VRAM или RAM)
                        # Этот эндпоинт доступен в новых версиях Ollama
                        status_resp = requests.get(f"{config.ollama_base_url}/api/ps")
                        if status_resp.status_code == 200:
                            models = status_resp.json().get("models", [])
                            for m in models:
                                if m.get("name") == config.ollama_model:
                                    vram = m.get("size_vram", 0) / (1024**3)
                                    logger.info(
                                        f"   ✅ Модель {config.ollama_model} готова. Использование VRAM: {vram:.2f} GB"
                                    )
                                    return
                        logger.info(
                            f"   ✅ Запрос на предпрогрев {config.ollama_model} выполнен"
                        )
                    else:
                        logger.warning(
                            f"   ⚠️ Ошибка при предпрогреве (код {response.status_code})"
                        )
                except Exception as e:
                    logger.warning(f"   ⚠️ Ошибка при предпрогреве: {e}")

            threading.Thread(target=warm_up, daemon=True).start()
        except Exception as e:
            logger.warning(f"   ⚠️ Не удалось запустить поток предпрогрева: {e}")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Ты — эксперт по технической документации. 
Твоя задача: давать КРАТКИЕ, максимально точные и конкретные ответы, основываясь ТОЛЬКО на предоставленных данных.

ОТВЕЧАЙ СТРОГО В ФОРМАТЕ JSON С КЛЮЧАМИ:
{{
  "answer": "краткий и точный ответ по существу (без лишних слов)",
  "source": "название документа и конкретный раздел/пункт",
  "quote": "точная цитата из текста, подтверждающая ответ"
}}

ПРАВИЛА:
1. Если информации в контексте недостаточно — в ключе "answer" отвечай "Информация не найдена".
2. Отвечай технически точно и МАКСИМАЛЬНО КРАТКО. Избегай длинных вступлений и подробных описаний, если вопрос этого не требует. Сразу переходи к фактам.
3. Если ответ можно дать одним предложением или короткой фразой — сделай это.
4. Не добавляй информацию от себя, используй только предоставленные фрагменты.
5. Используй ТОЛЬКО ключи "answer", "source", "quote". Ключи должны быть на английском.
6. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.""",
                ),
                ("human", "{context}\n\nВопрос: {question}"),
            ]
        )

    def _encode_image(self, image_path: str) -> Optional[str]:
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        try:
            full_path = Path(image_path)
            if not full_path.exists():
                full_path = Path(self.config.image_dir) / Path(image_path).name

            with open(full_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
                self.image_cache[str(full_path)] = base64_data
                return base64_data
        except Exception as e:
            logger.warning(f"   ⚠️ Ошибка при кодировании {image_path}: {e}")
            return None

    def _collect_images(
        self, docs: List[Document]
    ) -> tuple[List[Dict], Dict[str, int]]:
        """Возвращает список изображений для API и мапу path -> index (1-based)"""
        images = []
        path_to_index = {}

        for doc in docs:
            image_paths = doc.metadata.get("image_paths", [])[
                : self.config.max_images_per_chunk
            ]
            for path in image_paths:
                if len(images) >= self.config.max_total_images:
                    break
                if path in path_to_index:
                    continue

                base64_image = self._encode_image(path)
                if base64_image:
                    images.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        }
                    )
                    path_to_index[path] = len(images)

            if len(images) >= self.config.max_total_images:
                break

        return images, path_to_index

    async def index_documents(
        self, chunks: List[DocumentChunk], recreate: bool = False
    ) -> Dict[str, Any]:
        print("\n" + "=" * 60)
        print("📚 ИНДЕКСАЦИЯ ДОКУМЕНТОВ")
        print("=" * 60)

        main_documents = []
        table_documents = []

        for chunk in chunks:
            image_paths = chunk.metadata.get("image_paths", [])
            is_table = (
                chunk.metadata.get("is_table", False)
                or chunk.source_type == "docx_table"
            )

            doc = Document(
                page_content=chunk.content,
                metadata={
                    "document_name": chunk.metadata.get("document_name", ""),
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "total_chunks": chunk.metadata.get("total_chunks", 0),
                    "source_type": chunk.source_type,
                    "image_paths": image_paths,
                    "image_count": len(image_paths),
                    "has_images": len(image_paths) > 0,
                    "is_table": is_table,
                },
            )

            if is_table:
                table_documents.append(doc)
            else:
                main_documents.append(doc)

        print(
            f"\n1️⃣ Создано документов: основной поток - {len(main_documents)}, таблицы - {len(table_documents)}"
        )

        print("\n2️⃣ Настройка Qdrant...")

        async def prepare_collection(name, recreate_flag):
            if recreate_flag:
                try:
                    await self.async_qdrant_client.delete_collection(name)
                    logger.info(f"🗑️ Коллекция {name} удалена")
                except Exception:
                    pass

            collections = await self.async_qdrant_client.get_collections()
            exists = any(c.name == name for c in collections.collections)

            if not exists:
                test_embedding = await self.embeddings.aembed_query("test")

                await self.async_qdrant_client.create_collection(
                    collection_name=name,
                    vectors_config=rest_models.VectorParams(
                        size=len(test_embedding), distance=rest_models.Distance.COSINE
                    ),
                )
                logger.info(f"✅ Коллекция {name} создана")
            elif not recreate_flag:
                unique_docs = set(
                    c.metadata.get("document_name")
                    for c in chunks
                    if c.metadata.get("document_name")
                )
                if unique_docs:
                    logger.info(f"🔄 Очистка {name} перед обновлением...")
                    for doc_name in unique_docs:
                        for key in ["document_name", "metadata.document_name"]:
                            await self.async_qdrant_client.delete(
                                collection_name=name,
                                points_selector=Filter(
                                    must=[
                                        FieldCondition(
                                            key=key, match=MatchValue(value=doc_name)
                                        )
                                    ]
                                ),
                            )

        await prepare_collection(self.config.collection_name, recreate)
        await prepare_collection(self.config.tables_collection_name, recreate)
        await prepare_collection(self.config.summaries_collection_name, recreate)

        # Инициализируем vector_stores
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.collection_name,
                embedding=self.embeddings,
            )

        if self.tables_vector_store is None:
            self.tables_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.tables_collection_name,
                embedding=self.embeddings,
            )

        if self.summaries_vector_store is None:
            self.summaries_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.config.summaries_collection_name,
                embedding=self.embeddings,
            )

        # Загружаем документы
        async def upload_batch(v_store, docs, desc, collection_name):
            if not docs:
                return []
            print(f"\n4️⃣ Загрузка в {desc}...")
            doc_ids = []
            batch_size = 16 if self.actual_device.startswith("cuda") else 4

            for i in tqdm(range(0, len(docs), batch_size), desc=f"Загрузка ({desc})"):
                batch = docs[i : i + batch_size]
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]

                # Генерируем dense эмбеддинги
                dense_vectors = await self.embeddings.aembed_documents(texts)

                points = []
                import uuid

                for j in range(len(batch)):
                    point_id = str(uuid.uuid4())
                    vector_data = {"": dense_vectors[j]}

                    points.append(
                        rest_models.PointStruct(
                            id=point_id, vector=vector_data, payload=metadatas[j]
                        )
                    )
                    # Сохраняем текст в payload
                    points[-1].payload["page_content"] = texts[j]

                await self.async_qdrant_client.upsert(
                    collection_name=collection_name, points=points
                )
                doc_ids.extend([p.id for p in points])

                if self.actual_device.startswith("cuda"):
                    torch.cuda.empty_cache()
            return doc_ids

        main_ids = await upload_batch(
            self.vector_store,
            main_documents,
            "основная база",
            self.config.collection_name,
        )
        table_ids = await upload_batch(
            self.tables_vector_store,
            table_documents,
            "табличная база",
            self.config.tables_collection_name,
        )

        # 5️⃣ Создание и индексация двойных чанков (СУММАРИЗАЦИЯ)
        summary_ids = []
        if main_documents:
            print(
                "\n5️⃣ Создание суммаризаций для двойных чанков (с группировкой по документам)..."
            )

            # Группируем по документам
            docs_by_name = {}
            for doc in main_documents:
                name = doc.metadata.get("document_name", "unknown")
                if name not in docs_by_name:
                    docs_by_name[name] = []
                docs_by_name[name].append(doc)

            # Создаем задачи на суммаризацию
            tasks = []
            semaphore = asyncio.Semaphore(
                5
            )  # Максимум 5 параллельных запросов к Ollama

            async def process_pair(chunk1, chunk2=None, idx=0):
                async with semaphore:
                    if chunk2:
                        combined_content = (
                            f"{chunk1.page_content}\n\n{chunk2.page_content}"
                        )
                        combined_metadata = chunk1.metadata.copy()
                        combined_metadata["chunk_index"] = (
                            f"{chunk1.metadata.get('chunk_index', idx)}-{chunk2.metadata.get('chunk_index', idx + 1)}"
                        )
                        combined_metadata["is_double"] = True
                        combined_metadata["parent_chunk_ids"] = [
                            chunk1.metadata.get("chunk_id"),
                            chunk2.metadata.get("chunk_id"),
                        ]
                    else:
                        combined_content = chunk1.page_content
                        combined_metadata = chunk1.metadata.copy()
                        combined_metadata["is_double"] = True
                        combined_metadata["parent_chunk_ids"] = [
                            chunk1.metadata.get("chunk_id")
                        ]

                    logger.info(
                        f"   📝 Суммаризация: {combined_metadata.get('document_name')} (пара {idx // 2 + 1})"
                    )
                    summary_text = await self._summarize_content(combined_content)

                    return Document(
                        page_content=combined_content,
                        metadata={**combined_metadata, "summary": summary_text},
                    )

            for doc_name, chunks in docs_by_name.items():
                for i in range(0, len(chunks), 2):
                    c1 = chunks[i]
                    c2 = chunks[i + 1] if i + 1 < len(chunks) else None
                    tasks.append(process_pair(c1, c2, i))

            print(f"   🚀 Запуск {len(tasks)} задач суммаризации...")
            double_chunks_docs = await asyncio.gather(*tasks)

            # Специальный метод загрузки для суммаризаций

            # Специальный метод загрузки для суммаризаций (т.к. векторизуем summary, а сохраняем content)
            print("\n6️⃣ Загрузка суммаризаций в базу...")
            for i in tqdm(
                range(0, len(double_chunks_docs), 8), desc="Загрузка суммаризаций"
            ):
                batch = double_chunks_docs[i : i + 8]
                # Векторизуем СУММАРИЗАЦИИ
                summaries = [doc.metadata["summary"] for doc in batch]
                dense_vectors = await self.embeddings.aembed_documents(summaries)

                points = []
                import uuid

                for j in range(len(batch)):
                    point_id = str(uuid.uuid4())
                    points.append(
                        rest_models.PointStruct(
                            id=point_id,
                            vector={"": dense_vectors[j]},
                            payload={
                                **batch[j].metadata,
                                "page_content": batch[
                                    j
                                ].page_content,  # Сохраняем полный контент
                            },
                        )
                    )

                await self.async_qdrant_client.upsert(
                    collection_name=self.config.summaries_collection_name, points=points
                )
                summary_ids.extend([p.id for p in points])

        # Создаем retrievers
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.config.retriever_k}
        )
        self.tables_retriever = self.tables_vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.tables_retriever_k},
        )
        self.summaries_retriever = self.summaries_vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.summary_retriever_k},
        )

        stats = {
            "total_chunks": len(chunks),
            "main_chunks": len(main_documents),
            "table_chunks": len(table_documents),
            "main_uploaded": len(main_ids),
            "tables_uploaded": len(table_ids),
            "chunks_uploaded": len(main_ids) + len(table_ids),
        }

        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ ИНДЕКСАЦИИ")
        print("=" * 60)
        print(f"✓ Всего чанков: {stats['total_chunks']}")
        print(f"  - Основная база: {stats['main_uploaded']}")
        print(f"  - Табличная база: {stats['tables_uploaded']}")
        print("=" * 60)

        return stats

    async def _summarize_content(self, text: str) -> str:
        """Создает краткую суммаризацию текста для векторизации"""
        prompt = f"""Никаких вступлений! Выдавай ТОЛЬКО сухие факты.

ПРИМЕР:
Текст: "Инженер наносит клей ВК-27 при температуре 20 градусов и сушит 30 минут."
Результат: Клей ВК-27, нанесение при +20°С, сушка 30 мин.

ТЕКСТ ДЛЯ ОБРАБОТКИ:
{text}

Результат:"""
        try:
            # Маленькая быстрая модель для массовой суммаризации (не основная LLM)
            summary_model = os.getenv("SUMMARY_MODEL", "qwen3.5:4b")
            raw_summary = self._call_ollama_api(
                system_instr="Ты — сухой технический справочник. Твой ответ ВСЕГДА начинается сразу с фактов. Запрещено использовать любые вежливые слова, вступления или описания процесса.",
                user_content=prompt,
                model=summary_model,
                use_thinking=False,
                options={"num_predict": 300, "num_ctx": 4096, "temperature": 0.0},
            )

            return raw_summary.strip()
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при суммаризации: {e}")
            return text[:1000]  # Фолбэк на начало текста

    def _call_ollama_api(
        self,
        system_instr: str,
        user_content: str,
        images_base64: List[str] = None,
        response_format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        use_thinking: Optional[bool] = None,
    ) -> str:
        """Базовый метод для прямого вызова Ollama API с ретраями"""

        messages = [{"role": "system", "content": system_instr}]
        user_msg = {"role": "user", "content": user_content}
        if images_base64:
            user_msg["images"] = images_base64
        messages.append(user_msg)

        max_retries = 3
        retry_delay = 2
        last_err = None

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"📡 Запрос к Ollama API (попытка {attempt + 1}/{max_retries})..."
                )
                ollama_url = f"{self.config.ollama_base_url}/api/chat"
                payload = {
                    "model": model if model else self.config.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "think": use_thinking
                    if use_thinking is not None
                    else self.config.enable_thinking,
                    "options": {
                        "temperature": 0.0,
                        "num_ctx": self.config.ollama_num_ctx,
                        "seed": self.config.ollama_seed,
                        **(options or {}),
                    },
                }
                if response_format:
                    payload["format"] = response_format

                response = requests.post(
                    ollama_url, json=payload, timeout=self.config.ollama_timeout
                )
                response.raise_for_status()
                raw_data = response.json()
                content = raw_data.get("message", {}).get("content", "").strip()

                # Если "мышление" отключено, вырезаем блоки <thought>...</thought>
                if not self.config.enable_thinking:
                    # Регулярка для удаления блоков <thought>...</thought> и <thought> до конца (если не закрыто)
                    content = re.sub(
                        r"<thought>.*?</thought>", "", content, flags=re.DOTALL
                    )
                    content = re.sub(
                        r"<thought>.*", "", content, flags=re.DOTALL
                    ).strip()

                return content
            except (
                requests.exceptions.RequestException,
                requests.exceptions.Timeout,
            ) as err:
                last_err = err
                logger.warning(
                    f"⚠️ Ошибка при обращении к Ollama (попытка {attempt + 1}): {err}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise last_err
        return ""

    def _process_llm_response(self, raw_text: str) -> RAGResponse:
        """Парсинг и маппинг JSON ответа от LLM"""
        try:
            cleaned_text = raw_text
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.split("```json")[-1].split("```")[0].strip()
            elif "```" in cleaned_text:
                cleaned_text = cleaned_text.split("```")[-1].split("```")[0].strip()

            data = json.loads(cleaned_text)

            # Маппинг русских ключей в английские
            normalized_data = {}
            mapping = {
                "ответ": "answer",
                "answer": "answer",
                "источник": "source",
                "source": "source",
                "цитата": "quote",
                "quote": "quote",
                "обоснование": "answer",
            }

            for k, v in data.items():
                norm_k = mapping.get(k.lower())
                if norm_k:
                    normalized_data[norm_k] = v

            # Если мы не нашли ни одного стандартного ключа, берем самое длинное значение из словаря как ответ
            if "answer" not in normalized_data and data:
                # Фильтруем технические артефакты (числа, очень короткие строки)
                potential_answers = [
                    str(v)
                    for v in data.values()
                    if isinstance(v, (str, list)) and len(str(v)) > 10
                ]
                if potential_answers:
                    normalized_data["answer"] = max(potential_answers, key=len)
                else:
                    normalized_data["answer"] = raw_text

            # Обработка ответа, если он пришел в виде списка или словаря
            ans_val = normalized_data.get("answer", raw_text)
            if isinstance(ans_val, list):
                ans_val = ", ".join([str(v) for v in ans_val])
            elif isinstance(ans_val, dict):
                # Если ответ это просто плоский словарь, пытаемся вытащить значения
                ans_val = ", ".join([str(v) for v in ans_val.values()])

            # Подготавливаем результирующий объект
            return RAGResponse(
                answer=str(ans_val).strip("[]{}'\" "),
                source=str(normalized_data.get("source", "")),
                quote=str(normalized_data.get("quote", "")),
            )
        except Exception as parse_err:
            logger.warning(f"⚠️ Ошибка парсинга JSON: {parse_err}. RAW: {raw_text}")
            return RAGResponse(answer=raw_text, source="Ошибка парсинга", quote="")

    async def _get_response_for_chunks(
        self,
        docs_subset: List[Document],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> RAGResponse:
        """Получает ответ для подмножества чанков"""
        full_text_log = ""
        added_image_paths = set()
        path_to_index = {}

        # Индексация изображений для этого подмножества
        if self.config.use_images:
            images_list, path_to_index = self._collect_images(docs_subset)

        for i, doc in enumerate(docs_subset, 1):
            doc_name = doc.metadata.get("document_name", "")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            total = doc.metadata.get("total_chunks", 0)

            chunk_img_refs = []
            if self.config.use_images:
                image_paths = doc.metadata.get("image_paths", [])[
                    : self.config.max_images_per_chunk
                ]
                for path in image_paths:
                    if path in path_to_index:
                        idx = path_to_index[path]
                        chunk_img_refs.append(f"Изображение {idx}")
                        added_image_paths.add(path)

            img_ref_text = (
                f"\n🖼️ СВЯЗАННЫЕ ИЗОБРАЖЕНИЯ: {', '.join(chunk_img_refs)}"
                if chunk_img_refs
                else ""
            )

            chunk_text = f"""
[ЧАНК {i}]
📌 ДОКУМЕНТ: {doc_name}
🔢 ПОЗИЦИЯ: часть {chunk_idx}/{total}{img_ref_text}

ТЕКСТ ДОКУМЕНТА:
{doc.page_content}
{"-" * 50}
"""
            full_text_log += chunk_text

        images_base64 = []
        for path in added_image_paths:
            b64 = self._encode_image(path)
            if b64:
                images_base64.append(b64)

        system_instr = (
            system_prompt if system_prompt else self.prompt.messages[0].prompt.template
        )

        # Ограничение длины Промпта до 35 000 символов (чтобы Ollama не падала на огромных таблицах)
        # 16384 токенов ~ 45-60к символов. Мы берем с запасом.
        max_chars = 35000
        if len(full_text_log) > max_chars:
            logger.warning(
                f"⚠️ Текст контекста слишком большой ({len(full_text_log)} символов). Обрезаю до {max_chars}..."
            )
            full_text_log = (
                full_text_log[:max_chars]
                + "\n\n... [ДАННЫЕ ОБРЕЗАНЫ ИЗ-ЗА БОЛЬШОГО РАЗМЕРА ДЛЯ СТАБИЛЬНОСТИ LLM] ..."
            )

        user_content = f"{full_text_log}\n\nВопрос: {question}"

        raw_response = self._call_ollama_api(
            system_instr, user_content, images_base64, response_format="json"
        )
        return self._process_llm_response(raw_response)

    async def query(
        self, question: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.retriever:
            raise ValueError("Retriever не создан. Сначала выполните index_documents()")

        logger.info("\n🔍 Поиск фрагментов (векторный)...")

        # Генерируем dense эмбеддинг
        dense_vec = await self.embeddings.aembed_query(question)

        from langchain_core.documents import Document as LCDocument

        async def search(collection_name, k):
            res = await self.async_qdrant_client.query_points(
                collection_name=collection_name, query=dense_vec, limit=k
            )
            return [
                LCDocument(
                    page_content=item.payload.get("page_content", ""),
                    metadata={**item.payload, "score": item.score},
                )
                for item in res.points
            ]

        # Параллельный поиск по всем базам
        logger.info("   🔎 Ансамблевый поиск: синтез данных из 3-х источников")

        main_docs_task = search(self.config.collection_name, self.config.retriever_k)
        table_docs_task = search(
            self.config.tables_collection_name, self.config.tables_retriever_k
        )
        summary_docs_task = (
            search(
                self.config.summaries_collection_name, self.config.summary_retriever_k
            )
            if self.summaries_retriever
            else None
        )

        # Собираем документы
        if summary_docs_task:
            main_docs, table_docs, summary_docs = await asyncio.gather(
                main_docs_task, table_docs_task, summary_docs_task
            )
        else:
            main_docs, table_docs = await asyncio.gather(
                main_docs_task, table_docs_task
            )
            summary_docs = []

        # --- УМНАЯ ДЕДУПЛИКАЦИЯ ---
        covered_ids = set()
        for doc in summary_docs:
            parents = doc.metadata.get("parent_chunk_ids", [])
            if isinstance(parents, list):
                covered_ids.update(parents)

        # Фильтруем ТОЛЬКО для промпта LLM
        unique_main_docs = [
            d for d in main_docs if d.metadata.get("chunk_id") not in covered_ids
        ]

        if len(unique_main_docs) < len(main_docs):
            logger.info(
                f"   ✂️ Дедупликация для LLM: убрано {len(main_docs) - len(unique_main_docs)} чанков из промпта"
            )

        # В список источников для UI и сортировки берем ВСЁ БЕЗ ИСКЛЮЧЕНИЙ
        all_raw_docs = main_docs + table_docs + summary_docs
        all_retrieved_docs = sorted(
            all_raw_docs, key=lambda x: x.metadata.get("score", 0), reverse=True
        )

        # Логируем топ-3 источника (до дедупликации)
        for i, doc in enumerate(all_retrieved_docs[:3]):
            logger.info(
                f"   🔝 Топ-{i + 1} результат поиска: {doc.metadata.get('document_name')} (Score: {doc.metadata.get('score', 0):.4f})"
            )

        # Функция для получения промежуточного ответа
        async def get_sub_answer(docs, source_name):
            if not docs:
                return "Информация не найдена."
            logger.info(
                f"   🤖 Анализ слоя: {source_name} ({len(docs)} уникальных док.)..."
            )
            resp = await self._get_response_for_chunks(
                docs, question, system_prompt=system_prompt
            )
            return resp.answer

        # Шаг 1-3: Получаем 3 экспертных мнения (используем только УНИКАЛЬНЫЕ данные для main_docs)
        ans_main = await get_sub_answer(unique_main_docs, "Детали (одиночные чанки)")
        ans_summary = await get_sub_answer(
            summary_docs, "Смысловые блоки (двойные чанки)"
        )
        ans_tables = await get_sub_answer(table_docs, "Табличные данные")

        # Шаг 4: Интеллектуальный синтез
        logger.info("   🧠 СИНТЕЗ ФИНАЛЬНОГО ОТВЕТА...")

        # Инструкция для синтеза (баланс между краткостью и полнотой)
        synthesis_instr = "Ты — главный технический эксперт. Твоя задача: синтезировать точный и конкретный ответ на основе предоставленных данных. Избегай общих фраз, пиши по существу. Если есть противоречия, верь ТАБЛИЦАМ."
        if system_prompt:
            synthesis_instr += f"\nДополнительная установка: {system_prompt}"

        aggregation_prompt = f"""Вопрос: {question}

Перед тобой данные из трех аналитических слоев документа. Проведи их синтез.

ДАННЫЕ ДЛЯ СИНТЕЗА:
1) Информация из текста: 
{ans_main}

2) Информация из кратких выжимок: 
{ans_summary}

3) Информация из таблиц: 
{ans_tables}

ТРЕБОВАНИЯ:
- Сформируй единый, технически грамотный ответ.
- Используй списки только если это необходимо для перечисления параметров.
- Не добавляй вступлений («Основываясь на...») и заключений. Только факты.

Формат ответа JSON:
{{
  "answer": "твой синтезированный ответ",
  "source": "список всех упомянутых документов через запятую",
  "quote": "ключевая техническая цитата"
}}
"""

        start_time = time.time()
        final_raw = self._call_ollama_api(
            system_instr=synthesis_instr,
            user_content=aggregation_prompt,
            response_format="json",
        )
        final_res_obj = self._process_llm_response(final_raw)

        # Формируем список источников для UI (все найденные чанки)
        final_sources = []
        for doc in all_retrieved_docs:
            final_sources.append(
                {
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "document_name": doc.metadata.get("document_name", "Неизвестно"),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        elapsed = time.time() - start_time
        logger.info(f"✅ Синтез завершен за {elapsed:.1f} сек.")

        return {
            "answer": final_res_obj.answer,
            "source": final_res_obj.source,
            "quote": final_res_obj.quote,
            "sources": final_sources,
            "images_used": 0,
            "chunks_retrieved": len(all_retrieved_docs),
        }

    def _load_docs_data(self) -> List[Dict[str, Any]]:
        """Внутренний метод для получения данных о документах из Qdrant"""
        if not self.qdrant_client:
            return []
        all_docs = collections.defaultdict(int)
        offset = None
        while True:
            response = self.qdrant_client.scroll(
                collection_name=self.config.collection_name,
                limit=1000,
                with_payload=["document_name", "metadata.document_name"],
                with_vectors=False,
                offset=offset,
            )
            points, offset = response
            for point in points:
                p = point.payload
                name = (
                    p.get("document_name")
                    or p.get("metadata", {}).get("document_name")
                    or "Unknown"
                )
                all_docs[name] += 1
            if offset is None:
                break
        result = [{"name": name, "chunks": count} for name, count in all_docs.items()]
        return sorted(result, key=lambda x: x["name"])

    def list_indexed_documents_sync(self) -> List[Dict[str, Any]]:
        """Возвращает список документов из кэша или Qdrant (синхронно)"""
        # Кэширование на уровне сервиса
        if hasattr(self, "_docs_list_cache") and (
            time.time() - self._docs_list_cache_time < 10
        ):
            return self._docs_list_cache

        try:
            data = self._load_docs_data()
            self._docs_list_cache = data
            self._docs_list_cache_time = time.time()
            return data
        except Exception as e:
            logger.error(f"❌ Ошибка при получении списка (синхронно): {e}")
            return []

    async def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """Возвращает список документов (асинхронная обертка над синхронным методом)"""
        return await anyio.to_thread.run_sync(self.list_indexed_documents_sync)

    async def delete_document(self, doc_name: str) -> bool:
        """Удаляет все чанки документа из Qdrant (асинхронно)"""
        if not self.async_qdrant_client:
            return False

        try:
            logger.info(f"🗑️ Удаление документа из баз: {doc_name}")

            filter_obj = rest_models.Filter(
                should=[
                    rest_models.FieldCondition(
                        key="document_name",
                        match=rest_models.MatchValue(value=doc_name),
                    ),
                    rest_models.FieldCondition(
                        key="metadata.document_name",
                        match=rest_models.MatchValue(value=doc_name),
                    ),
                ]
            )

            # Удаляем из всех возможных коллекций
            collections = [
                self.config.collection_name,
                self.config.tables_collection_name,
                self.config.summaries_collection_name,
            ]
            for col in collections:
                try:
                    await self.async_qdrant_client.delete(
                        collection_name=col, points_selector=filter_obj
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка при удалении из {col}: {e}")

            # Сбрасываем кэш списка документов
            if hasattr(self, "_docs_list_cache_time"):
                self._docs_list_cache_time = 0

            return True
        except Exception as e:
            logger.error(f"❌ Ошибка при удалении документа {doc_name}: {e}")
            return False

    def delete_document_sync(self, doc_name: str) -> bool:
        """Удаляет все чанки документа (синхронная обертка)"""
        return asyncio.run(self.delete_document(doc_name))
