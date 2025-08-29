from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
from core.services.llm_service import LLMFactory
from core.services.cache_service import CacheService, get_cache_service
from core.search.hybrid_search import HybridSearchEngine
from core.search.bm25_search import BM25SearchEngine
from core.search.vector_search import VectorSearchEngine
from core.processors.document_processor import DocumentProcessor, RAGApplication
from config.settings import HybridSearchConfig

# New clean architecture imports
from core.services.clean_document_processor import (
    CleanDocumentProcessor, VectorDocumentRepository, 
    BM25IndexBuilder, CleanProcessingOrchestrator
)
from core.services.clean_search_engines import CleanSearchEngineFactory
from core.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


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


# New immutable container architecture to replace global state pattern
@dataclass(frozen=True)
class ImmutableAppContainer:
    """
    Immutable container holding fully initialized services.
    Replaces the mutable global state pattern with proper dependency injection.
    """
    data_dir: str
    pg_dsn: str
    cache_service: CacheService
    llm_factory: LLMFactory
    bm25_engine: BM25SearchEngine
    vector_engine: VectorSearchEngine
    hybrid_engine: HybridSearchEngine
    
    def is_ready(self) -> bool:
        """Check if all required services are properly initialized."""
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
        stats = self.cache_service.get_cache_stats()
        stats["cache_available"] = self.cache_service.is_available()
        return stats
    
    # New clean architecture factory methods
    def create_clean_processor(self) -> CleanDocumentProcessor:
        """Create clean document processor with proper separation of concerns."""
        return CleanDocumentProcessor()
    
    def create_vector_repository(self) -> VectorDocumentRepository:
        """Create vector document repository."""
        vector_store = VectorStore()
        return VectorDocumentRepository(vector_store)
    
    def create_bm25_index_builder(self) -> BM25IndexBuilder:
        """Create BM25 index builder."""
        return BM25IndexBuilder(self.bm25_engine)
    
    def create_processing_orchestrator(self) -> CleanProcessingOrchestrator:
        """Create complete processing orchestrator with all components."""
        processor = self.create_clean_processor()
        repository = self.create_vector_repository()
        index_builder = self.create_bm25_index_builder()
        return CleanProcessingOrchestrator(processor, repository, index_builder)
    
    def create_search_engine_factory(self) -> CleanSearchEngineFactory:
        """Create search engine factory for clean architecture."""
        vector_store = VectorStore()
        return CleanSearchEngineFactory(vector_store, self.bm25_engine)


def create_immutable_container(data_dir: str, pg_dsn: str) -> ImmutableAppContainer:
    """
    Factory function to create fully initialized immutable container.
    All services are initialized upfront, eliminating mutable state issues.
    """
    try:
        logger.info("Creating immutable app container with all services")
        
        # Initialize core services
        cache_service = get_cache_service()
        llm_factory = LLMFactory()
        
        # Initialize search engines
        vector_engine = VectorSearchEngine()
        bm25_engine = BM25SearchEngine()
        
        # Build empty BM25 index for initialization (will be populated during /init)
        # This ensures the container reports as "ready" even before document processing
        import pandas as pd
        empty_df = pd.DataFrame(columns=['uuid_chunk', 'chunk_enriched', 'file_name', 'keywords', 'metadata'])
        try:
            bm25_engine.build_index(empty_df)
        except ValueError:
            # Expected when no documents - BM25 will be rebuilt during /init
            logger.debug("BM25 engine initialized with empty index")
        
        # Initialize hybrid engine with default config
        config = HybridSearchConfig()
        hybrid_engine = HybridSearchEngine(bm25_engine, vector_engine, config)
        
        container = ImmutableAppContainer(
            data_dir=data_dir,
            pg_dsn=pg_dsn,
            cache_service=cache_service,
            llm_factory=llm_factory,
            bm25_engine=bm25_engine,
            vector_engine=vector_engine,
            hybrid_engine=hybrid_engine,
        )
        
        logger.info(f"Immutable container created successfully, ready: {container.is_ready()}")
        return container
        
    except Exception as e:
        logger.error(f"Failed to create immutable container: {e}")
        raise RuntimeError(f"Container initialization failed: {e}") from e


def create_test_container(data_dir: str = "./tests/fixtures/data", 
                         pg_dsn: str = "postgresql://test:test@localhost:5432/test") -> ImmutableAppContainer:
    """
    Factory function for creating isolated test containers.
    Used in testing to provide clean, isolated dependencies.
    """
    return create_immutable_container(data_dir, pg_dsn)
