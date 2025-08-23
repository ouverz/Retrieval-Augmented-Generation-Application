from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from core.services.llm_service import LLMFactory
from core.services.cache_service import CacheService, get_cache_service
from core.search.hybrid_search import HybridSearchEngine
from core.search.bm25_search import BM25SearchEngine
from core.search.vector_search import VectorSearchEngine
from core.processors.document_processor import DocumentProcessor, RAGApplication


@dataclass
class AppContainer:
    # runtime config
    data_dir: str
    pg_dsn: str

    # lazily populated after /init
    doc_processor: Optional[DocumentProcessor] = None
    rag_app: Optional[RAGApplication] = None
    bm25_engine: Optional[BM25SearchEngine] = None
    vector_engine: Optional[VectorSearchEngine] = None
    hybrid_engine: Optional[HybridSearchEngine] = None
    llm_factory: Optional[LLMFactory] = None
    cache_service: Optional[CacheService] = None

    def __post_init__(self):
        """Initialize cache service on container creation."""
        if self.cache_service is None:
            self.cache_service = get_cache_service()

    def is_ready(self) -> bool:
        return (
            self.hybrid_engine is not None 
            and self.llm_factory is not None 
            and self.bm25_engine is not None 
            and self.bm25_engine.retriever is not None
            and self.vector_engine is not None
            and self.cache_service is not None
        )
    
    def get_cache_health(self) -> dict:
        """Get cache service health and statistics."""
        if self.cache_service is None:
            return {"error": "Cache service not initialized"}
        
        stats = self.cache_service.get_cache_stats()
        stats["cache_available"] = self.cache_service.is_available()
        return stats
