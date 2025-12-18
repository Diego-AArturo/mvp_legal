from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphSearchResult:
    id: str
    tipo: Optional[str] = None
    nombre: Optional[str] = None
    texto_relevante: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    conceptos_clave: List[Any] = field(default_factory=list)
    normas_citadas: List[Any] = field(default_factory=list)
    entidades_mencionadas: List[Any] = field(default_factory=list)
    documento: Optional[str] = None

    def model_dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tipo": self.tipo,
            "nombre": self.nombre,
            "texto_relevante": self.texto_relevante,
            "score": self.score,
            "metadata": self.metadata,
            "conceptos_clave": self.conceptos_clave,
            "normas_citadas": self.normas_citadas,
            "entidades_mencionadas": self.entidades_mencionadas,
            "documento": self.documento,
        }


@dataclass
class GraphSearchResponse:
    results: List[GraphSearchResult] = field(default_factory=list)
    total_found: int = 0
    execution_time: float = 0.0
    search_strategy: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_results(self) -> bool:
        return bool(self.total_found)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "results": [r.model_dump() for r in self.results],
            "total_found": self.total_found,
            "execution_time": self.execution_time,
            "search_strategy": self.search_strategy,
            "metadata": self.metadata,
        }

