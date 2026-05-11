"""Асинхронный aiohttp-клиент к Triton."""

from __future__ import annotations

import aiohttp
from typing import List

TRITON_MODEL = "bge_model"
MAX_LENGTH = 512


class TritonInferClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def infer(
        self,
        input_ids: List[int],
        attention_mask: List[int],
    ) -> List[float]:
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [1, MAX_LENGTH],
                    "datatype": "INT64",
                    "data": input_ids,
                },
                {
                    "name": "attention_mask",
                    "shape": [1, MAX_LENGTH],
                    "datatype": "INT64",
                    "data": attention_mask,
                },
            ]
        }
        url = f"{self._base_url}/v2/models/{TRITON_MODEL}/infer"
        session = await self._get_session()
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
        out = data.get("outputs", [])
        if not out or out[0].get("name") != "output":
            raise ValueError("Unexpected Triton response shape")
        return out[0]["data"]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
