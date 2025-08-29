"""
Document processing interfaces for clean architecture separation.
Defines contracts between processing, persistence, and search layers.
"""

from __future__ import annotations
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ProcessedDocument:
    """Represents a processed document with all required metadata."""
    uuid_chunk: str
    content_raw: str
    content_enriched: str
    embeddings: List[float]
    metadata: Dict[str, Any]
    keywords: List[str]
    file_path: str
    chunk_index: int


@dataclass  
class ProcessingResult:
    """Result of document processing operation."""
    documents: List[ProcessedDocument]
    total_chunks: int
    processing_time_seconds: float
    errors: List[str]
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert processed documents to DataFrame format."""
        if not self.documents:
            return pd.DataFrame()
            
        rows = []
        for doc in self.documents:
            rows.append({
                'uuid_chunk': doc.uuid_chunk,
                'chunk_text': doc.content_raw,
                'chunk_enriched': doc.content_enriched,
                'embeddings': doc.embeddings,
                'metadata': doc.metadata,
                'keywords': doc.keywords,
                'file_name': Path(doc.file_path).name,
            })
        
        return pd.DataFrame(rows)


class DocumentProcessor(ABC):
    """Abstract interface for document processing operations."""
    
    @abstractmethod
    def process_file(self, file_path: str) -> ProcessingResult:
        """Process a single document file."""
        pass
    
    @abstractmethod
    def process_directory(self, directory: str) -> ProcessingResult:
        """Process all documents in a directory.""" 
        pass


class DocumentRepository(ABC):
    """Abstract interface for document persistence operations."""
    
    @abstractmethod
    def save_documents(self, documents: List[ProcessedDocument]) -> bool:
        """Save processed documents to storage."""
        pass
    
    @abstractmethod
    def get_all_documents(self) -> pd.DataFrame:
        """Retrieve all documents as DataFrame."""
        pass
    
    @abstractmethod
    def clear_documents(self) -> bool:
        """Clear all documents from storage."""
        pass


class SearchIndexBuilder(ABC):
    """Abstract interface for building search indices."""
    
    @abstractmethod
    def build_vector_index(self, documents_df: pd.DataFrame) -> bool:
        """Build vector search index from documents."""
        pass
    
    @abstractmethod 
    def build_bm25_index(self, documents_df: pd.DataFrame) -> bool:
        """Build BM25 search index from documents."""
        pass
    
    @abstractmethod
    def build_hybrid_index(self, documents_df: pd.DataFrame) -> bool:
        """Build hybrid search index combining vector and BM25."""
        pass


class ProcessingOrchestrator(ABC):
    """Abstract interface for orchestrating the complete processing pipeline."""
    
    @abstractmethod
    def process_and_index(self, directory: str) -> ProcessingResult:
        """Complete pipeline: process documents and build all search indices."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if all components are ready for search operations."""
        pass