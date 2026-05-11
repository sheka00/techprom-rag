import aiohttp
import requests
import logging
from typing import List
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class TritonEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str,
        triton_url: str,
        triton_model: str = "bge_model",
        encoder_url: str = None,
        batch_size: int = 32,
    ):
        if not encoder_url:
            raise ValueError("Encoder service URL must be provided in this lite mode.")

        logger.info(
            f"🚀 Инициализация прокси-клиента TritonEmbeddings (Encoder: {encoder_url})"
        )
        self.encoder_url = encoder_url
        self.batch_size = batch_size
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _infer_remote_async(self, texts: List[str]) -> List[List[float]]:
        session = await self._get_session()
        payload = {"query": texts, "batch_size": self.batch_size}
        try:
            async with session.post(self.encoder_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.error(f"❌ Ошибка асинхронного запроса к Encoder Service: {e}")
            raise

    def _infer_remote_sync(self, texts: List[str]) -> List[List[float]]:
        payload = {"query": texts, "batch_size": self.batch_size}
        try:
            response = requests.post(self.encoder_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Ошибка синхронного запроса к Encoder Service: {e}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._infer_remote_async(texts)

    async def aembed_query(self, text: str) -> List[float]:
        res = await self._infer_remote_async([text])
        return res[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Синхронная реализация через requests (используется LangChain при инициализации)"""
        return self._infer_remote_sync(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
