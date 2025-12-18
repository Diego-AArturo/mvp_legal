from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VectorSearchResult:
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_id: Optional[str] = None

    def model_dump(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
            "source_id": self.source_id,
        }

