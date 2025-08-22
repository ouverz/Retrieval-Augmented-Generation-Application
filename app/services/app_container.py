from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from src.services.llm_factory import LLMFactory
from src.HybridSearchEngine import HybridSearchEngine
from src.BM25SearchEngine import BM25SearchEngine
from src.VectorSearchEngine import VectorSearchEngine
from src.Processing_Documents import DocumentProcessor, RAGApplication


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

    def is_ready(self) -> bool:
        return (
            self.hybrid_engine is not None 
            and self.llm_factory is not None 
            and self.bm25_engine is not None 
            and self.bm25_engine.retriever is not None
            and self.vector_engine is not None
        )
