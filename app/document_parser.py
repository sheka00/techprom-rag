from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    source_type: str = "docx"
    chunk_id: str = ""
