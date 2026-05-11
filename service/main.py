"""Микросервис: текст → токенизация → Triton → вектор. Эндпоинты: /health, /encode, /get_vector_dim."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, List, Optional, Union

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from transformers import AutoTokenizer

from triton_backend import MAX_LENGTH, TritonInferClient

MODEL_NAME = os.environ.get("MODEL_NAME", "deepvk/USER-bge-m3")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
TRITON_URL_ENV = "TRITON_URL"


class EncodeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Union[str, List[str]]
    prefix: Optional[str] = None
    batch_size: int = 32


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    triton_base_url = os.environ.get(TRITON_URL_ENV, "http://localhost:8000")
    triton_client = TritonInferClient(triton_base_url)
    app.state.tokenizer = tokenizer
    app.state.triton_client = triton_client
    yield
    await triton_client.close()


app = FastAPI(title="BGE Encoder", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/get_vector_dim")
async def get_vector_dim() -> int:
    return EMBEDDING_DIM


@app.post("/encode")
async def encode(request: Request, body: EncodeBody) -> Any:
    tokenizer = request.app.state.tokenizer
    client = request.app.state.triton_client

    texts: List[str] = [body.query] if isinstance(body.query, str) else list(body.query)
    prefix = (body.prefix or "").strip()
    if prefix:
        texts = [prefix + " " + t.strip() if t.strip() else t for t in texts]

    batch_size = max(1, min(body.batch_size, 64))

    def tokenize_one(text: str) -> tuple[List[int], List[int]]:
        enc = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        return enc["input_ids"], enc["attention_mask"]

    async def infer_one(input_ids: List[int], attention_mask: List[int]) -> List[float]:
        return await client.infer(input_ids, attention_mask)

    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokenized = [tokenize_one(t) for t in batch_texts]
        tasks = [infer_one(ids, mask) for ids, mask in tokenized]
        batch_embeddings = await asyncio.gather(*tasks)
        embeddings.extend(batch_embeddings)

    return embeddings
