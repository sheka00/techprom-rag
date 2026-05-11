import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:14b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    enable_thinking: bool = os.getenv("ENABLE_THINKING", "True").lower() == "true"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "deepvk/USER-bge-m3")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cuda")
    triton_url: str = os.getenv("TRITON_URL", "http://triton:8000")
    triton_model: str = os.getenv("TRITON_MODEL", "bge_model")
    encoder_url: str = os.getenv("ENCODER_URL", "http://encoder:8000/encode")
    use_triton: bool = os.getenv("USE_TRITON", "True").lower() == "true"
    qdrant_path: Optional[str] = os.getenv("QDRANT_PATH")
    qdrant_url: Optional[str] = os.getenv("QDRANT_URL", "http://qdrant:6333")
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_ti_docs")
    tables_collection_name: str = os.getenv("TABLES_COLLECTION_NAME", "rag_ti_tables")
    summaries_collection_name: str = os.getenv(
        "SUMMARIES_COLLECTION_NAME", "rag_ti_summaries"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    image_dir: str = os.getenv("IMAGE_DIR", "docx_images")
    use_images: bool = os.getenv("USE_IMAGES", "False").lower() == "true"
    max_images_per_chunk: int = int(os.getenv("MAX_IMAGES_PER_CHUNK", "10"))
    max_total_images: int = int(os.getenv("MAX_TOTAL_IMAGES", "5"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "20"))
    summary_retriever_k: int = int(os.getenv("SUMMARY_RETRIEVER_K", "10"))
    tables_retriever_k: int = int(os.getenv("TABLES_RETRIEVER_K", "5"))
    ollama_seed: int = int(os.getenv("OLLAMA_SEED", "42"))
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "16000"))
    debug_dir: str = os.getenv("DEBUG_DIR", "data/debug_chunks")
