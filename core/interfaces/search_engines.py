"""
Search engine interfaces for clean separation of search operations.
Defines contracts for different types of search engines.
"""

from __future__ import annotations
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document


@dataclass
class SearchResult:
    """Standardized search result format."""
    query: str
    documents: List[Document]
    total_results: int
    processing_time_ms: int
    metadata: Dict[str, Any]


class SearchEngine(ABC):
    """Abstract base interface for all search engines."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> SearchResult:
        """Perform search and return standardized results."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if search engine is ready for queries."""
        pass


class VectorSearchEngine(SearchEngine):
    """Interface for vector/semantic search operations only."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10, 
              metadata_filter: Optional[Dict] = None) -> SearchResult:
        """Perform vector similarity search."""
        pass


class KeywordSearchEngine(SearchEngine):
    """Interface for keyword/BM25 search operations only."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> SearchResult:
        """Perform keyword-based search."""
        pass


class HybridSearchEngine(SearchEngine):
    """Interface for hybrid search combining multiple engines."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10, 
              vector_weight: Optional[float] = None) -> SearchResult:
        """Perform hybrid search with configurable weighting."""
        pass
    
    @abstractmethod
    def get_component_engines(self) -> Dict[str, SearchEngine]:
        """Get underlying search engines for inspection."""
        pass


class SearchEngineFactory(ABC):
    """Abstract factory for creating search engines."""
    
    @abstractmethod
    def create_vector_engine(self) -> VectorSearchEngine:
        """Create vector search engine instance."""
        pass
    
    @abstractmethod
    def create_keyword_engine(self) -> KeywordSearchEngine:
        """Create keyword search engine instance."""
        pass
    
    @abstractmethod
    def create_hybrid_engine(self, vector_engine: VectorSearchEngine,
                           keyword_engine: KeywordSearchEngine) -> HybridSearchEngine:
        """Create hybrid search engine instance."""
        pass