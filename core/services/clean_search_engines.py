"""
Clean search engine implementations following the new interfaces.
Separates search operations from processing responsibilities.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Any
from langchain.schema import Document

from core.interfaces.search_engines import (
    SearchResult, VectorSearchEngine, KeywordSearchEngine, 
    HybridSearchEngine, SearchEngineFactory
)
from core.database.vector_store import VectorStore
from core.search.bm25_search import BM25SearchEngine

logger = logging.getLogger(__name__)


class CleanVectorSearchEngine(VectorSearchEngine):
    """Clean vector search engine - pure search operations."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def search(self, query: str, top_k: int = 10, 
              metadata_filter: Optional[Dict] = None) -> SearchResult:
        """Perform vector similarity search."""
        start_time = time.time()
        
        try:
            # Perform vector search
            search_kwargs = {}
            if metadata_filter:
                search_kwargs['metadata_filter'] = metadata_filter
                
            results_df = self.vector_store.search(
                query_text=query,
                limit=top_k,
                **search_kwargs
            )
            
            # Convert results to Document objects
            documents = []
            for _, row in results_df.iterrows():
                doc = Document(
                    page_content=row['content'],
                    metadata={
                        'id': row['id'],
                        'distance': row['distance'],
                        **{k: v for k, v in row.items() if k not in ['content', 'id', 'distance', 'embedding']}
                    }
                )
                documents.append(doc)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SearchResult(
                query=query,
                documents=documents,
                total_results=len(documents),
                processing_time_ms=processing_time,
                metadata={'search_type': 'vector', 'metadata_filter': metadata_filter}
            )
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            processing_time = int((time.time() - start_time) * 1000)
            return SearchResult(
                query=query,
                documents=[],
                total_results=0,
                processing_time_ms=processing_time,
                metadata={'error': str(e), 'search_type': 'vector'}
            )
    
    def is_ready(self) -> bool:
        """Check if vector search engine is ready."""
        # Basic check - could be enhanced
        return self.vector_store is not None


class CleanKeywordSearchEngine(KeywordSearchEngine):
    """Clean BM25 keyword search engine - pure search operations."""
    
    def __init__(self, bm25_engine: BM25SearchEngine):
        self.bm25_engine = bm25_engine
        
    def search(self, query: str, top_k: int = 10) -> SearchResult:
        """Perform keyword-based search."""
        start_time = time.time()
        
        try:
            if not self.is_ready():
                raise RuntimeError("BM25 engine not ready - index not built")
            
            # Perform BM25 search
            results_df = self.bm25_engine.search(query, top_k=top_k)
            
            # Convert results to Document objects
            documents = []
            for _, row in results_df.iterrows():
                doc = Document(
                    page_content=row['chunk_enriched'],  # Use enriched content
                    metadata={
                        'id': row['uuid_chunk'],
                        'score': row['bm25_score'],
                        'file_name': row.get('file_name', ''),
                        'chunk_index': row.get('chunk_index', 0)
                    }
                )
                documents.append(doc)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SearchResult(
                query=query,
                documents=documents,
                total_results=len(documents),
                processing_time_ms=processing_time,
                metadata={'search_type': 'bm25'}
            )
            
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            processing_time = int((time.time() - start_time) * 1000)
            return SearchResult(
                query=query,
                documents=[],
                total_results=0,
                processing_time_ms=processing_time,
                metadata={'error': str(e), 'search_type': 'bm25'}
            )
    
    def is_ready(self) -> bool:
        """Check if BM25 search engine is ready."""
        return (self.bm25_engine is not None and 
                self.bm25_engine.retriever is not None)


class CleanHybridSearchEngine(HybridSearchEngine):
    """Clean hybrid search engine combining vector and keyword search."""
    
    def __init__(
        self, 
        vector_engine: VectorSearchEngine,
        keyword_engine: KeywordSearchEngine,
        default_vector_weight: float = 0.6
    ):
        self.vector_engine = vector_engine
        self.keyword_engine = keyword_engine
        self.default_vector_weight = default_vector_weight
        
    def search(self, query: str, top_k: int = 10, 
              vector_weight: Optional[float] = None) -> SearchResult:
        """Perform hybrid search with configurable weighting."""
        start_time = time.time()
        
        try:
            # Use provided weight or default
            weight = vector_weight if vector_weight is not None else self.default_vector_weight
            keyword_weight = 1.0 - weight
            
            # Perform both searches
            vector_result = self.vector_engine.search(query, top_k=top_k)
            keyword_result = self.keyword_engine.search(query, top_k=top_k)
            
            # Combine results using document-level fusion
            combined_docs = self._document_level_fusion(
                vector_result.documents,
                keyword_result.documents,
                weight,
                keyword_weight
            )
            
            # Limit to top_k results
            combined_docs = combined_docs[:top_k]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SearchResult(
                query=query,
                documents=combined_docs,
                total_results=len(combined_docs),
                processing_time_ms=processing_time,
                metadata={
                    'search_type': 'hybrid',
                    'vector_weight': weight,
                    'keyword_weight': keyword_weight,
                    'vector_results': len(vector_result.documents),
                    'keyword_results': len(keyword_result.documents)
                }
            )
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            processing_time = int((time.time() - start_time) * 1000)
            return SearchResult(
                query=query,
                documents=[],
                total_results=0,
                processing_time_ms=processing_time,
                metadata={'error': str(e), 'search_type': 'hybrid'}
            )
    
    def get_component_engines(self) -> Dict[str, Any]:
        """Get underlying search engines for inspection."""
        return {
            'vector_engine': self.vector_engine,
            'keyword_engine': self.keyword_engine
        }
    
    def is_ready(self) -> bool:
        """Check if both component engines are ready."""
        return (self.vector_engine.is_ready() and 
                self.keyword_engine.is_ready())
    
    def _document_level_fusion(
        self, 
        vector_docs: List[Document], 
        keyword_docs: List[Document],
        vector_weight: float,
        keyword_weight: float
    ) -> List[Document]:
        """Perform document-level fusion of search results."""
        # Create content hash map for document matching
        doc_scores = {}
        
        # Process vector results
        for doc in vector_docs:
            content_hash = hash(doc.page_content)
            vector_score = 1.0 / (1.0 + doc.metadata.get('distance', 0))
            doc_scores[content_hash] = {
                'document': doc,
                'vector_score': vector_score,
                'keyword_score': 0.0,
                'content_hash': content_hash
            }
        
        # Process keyword results and match with vector results
        for doc in keyword_docs:
            content_hash = hash(doc.page_content)
            keyword_score = doc.metadata.get('score', 0)
            
            if content_hash in doc_scores:
                # Document found in both results - update keyword score
                doc_scores[content_hash]['keyword_score'] = keyword_score
            else:
                # Document only in keyword results
                doc_scores[content_hash] = {
                    'document': doc,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score,
                    'content_hash': content_hash
                }
        
        # Calculate combined scores and sort
        scored_docs = []
        for entry in doc_scores.values():
            combined_score = (
                entry['vector_score'] * vector_weight + 
                entry['keyword_score'] * keyword_weight
            )
            
            # Update document metadata with fusion information
            doc = entry['document']
            doc.metadata.update({
                'hybrid_score': combined_score,
                'vector_score': entry['vector_score'],
                'keyword_score': entry['keyword_score'],
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight
            })
            
            scored_docs.append((combined_score, doc))
        
        # Sort by combined score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs]


class CleanSearchEngineFactory(SearchEngineFactory):
    """Factory for creating clean search engine instances."""
    
    def __init__(self, vector_store: VectorStore, bm25_engine: BM25SearchEngine):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine
        
    def create_vector_engine(self) -> VectorSearchEngine:
        """Create vector search engine instance."""
        return CleanVectorSearchEngine(self.vector_store)
    
    def create_keyword_engine(self) -> KeywordSearchEngine:
        """Create keyword search engine instance."""
        return CleanKeywordSearchEngine(self.bm25_engine)
    
    def create_hybrid_engine(
        self, 
        vector_engine: VectorSearchEngine,
        keyword_engine: KeywordSearchEngine,
        vector_weight: float = 0.6
    ) -> HybridSearchEngine:
        """Create hybrid search engine instance."""
        return CleanHybridSearchEngine(
            vector_engine, 
            keyword_engine, 
            vector_weight
        )