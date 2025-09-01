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
              vector_weight: Optional[float] = None,
              rrf_k: Optional[int] = None) -> SearchResult:
        """Perform hybrid search using Reciprocal Rank Fusion (RRF) with comprehensive logging."""
        start_time = time.time()
        effective_rrf_k = rrf_k if rrf_k is not None else 60
        
        # Log search initiation with structured data
        logger.info("RRF hybrid search initiated", extra={
            "query_length": len(query),
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "top_k": top_k,
            "rrf_k_value": effective_rrf_k,
            "search_type": "hybrid_rrf",
            "fusion_method": "reciprocal_rank_fusion"
        })
        
        try:
            # Keep weight parameters for interface compatibility (not used in RRF)
            weight = vector_weight if vector_weight is not None else self.default_vector_weight
            keyword_weight = 1.0 - weight
            
            # Perform both searches with individual timing
            vector_start = time.time()
            vector_result = self.vector_engine.search(query, top_k=top_k)
            vector_time_ms = int((time.time() - vector_start) * 1000)
            
            keyword_start = time.time()
            keyword_result = self.keyword_engine.search(query, top_k=top_k)
            keyword_time_ms = int((time.time() - keyword_start) * 1000)
            
            # Log component search performance
            logger.debug("Component engine performance", extra={
                "vector_search_time_ms": vector_time_ms,
                "keyword_search_time_ms": keyword_time_ms,
                "vector_results": len(vector_result.documents),
                "keyword_results": len(keyword_result.documents),
                "query_preview": query[:50] + "..." if len(query) > 50 else query
            })
            
            # Combine results using RRF-based document-level fusion with timing
            fusion_start = time.time()
            combined_docs = self._document_level_fusion(
                vector_result.documents,
                keyword_result.documents,
                weight,  # Kept for compatibility but not used in RRF
                keyword_weight,  # Kept for compatibility but not used in RRF
                effective_rrf_k,
                query  # Pass query for logging context
            )
            fusion_time_ms = int((time.time() - fusion_start) * 1000)
            
            # Limit to top_k results
            combined_docs = combined_docs[:top_k]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Calculate RRF metrics and engine agreement
            rrf_metrics = self._calculate_rrf_metrics(combined_docs)
            
            # Log comprehensive search completion with RRF analytics
            logger.info("RRF hybrid search completed", extra={
                "total_processing_time_ms": processing_time,
                "fusion_processing_time_ms": fusion_time_ms,
                "vector_search_time_ms": vector_time_ms,
                "keyword_search_time_ms": keyword_time_ms,
                "input_vector_results": len(vector_result.documents),
                "input_keyword_results": len(keyword_result.documents),
                "final_results": len(combined_docs),
                "rrf_k_value": effective_rrf_k,
                **rrf_metrics
            })
            
            # Enhanced metadata with RRF-specific information
            enhanced_metadata = {
                'search_type': 'hybrid_rrf',
                'fusion_method': 'reciprocal_rank_fusion',
                'rrf_k_value': effective_rrf_k,
                'vector_results': len(vector_result.documents),
                'keyword_results': len(keyword_result.documents),
                'fusion_time_ms': fusion_time_ms,
                'vector_search_time_ms': vector_time_ms,
                'keyword_search_time_ms': keyword_time_ms,
                **rrf_metrics
            }
            
            return SearchResult(
                query=query,
                documents=combined_docs,
                total_results=len(combined_docs),
                processing_time_ms=processing_time,
                metadata=enhanced_metadata
            )
            
        except Exception as e:
            logger.error("RRF hybrid search failed", extra={
                "error": str(e),
                "query_preview": query[:100] + "..." if len(query) > 100 else query,
                "rrf_k_value": effective_rrf_k,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }, exc_info=True)
            processing_time = int((time.time() - start_time) * 1000)
            return SearchResult(
                query=query,
                documents=[],
                total_results=0,
                processing_time_ms=processing_time,
                metadata={
                    'error': str(e), 
                    'search_type': 'hybrid_rrf',
                    'rrf_k_value': effective_rrf_k
                }
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
        keyword_weight: float,
        rrf_k: int = 60,
        query: str = ""
    ) -> List[Document]:
        """Perform RRF-based fusion of search results with detailed logging."""
        fusion_start = time.time()
        
        logger.debug("Starting RRF document fusion", extra={
            "vector_docs_count": len(vector_docs),
            "keyword_docs_count": len(keyword_docs),
            "rrf_k_parameter": rrf_k,
            "query_preview": query[:50] + "..." if len(query) > 50 else query
        })
        # Create content hash map for document matching and rank tracking
        doc_data = {}
        
        # Process vector results - assign ranks based on search order
        for rank, doc in enumerate(vector_docs, 1):
            content_hash = hash(doc.page_content)
            vector_score = 1.0 / (1.0 + doc.metadata.get('distance', 0))
            doc_data[content_hash] = {
                'document': doc,
                'vector_rank': rank,
                'keyword_rank': None,
                'vector_score': vector_score,
                'keyword_score': 0.0,
                'content_hash': content_hash
            }
        
        # Process keyword results - assign ranks and match with vector results
        for rank, doc in enumerate(keyword_docs, 1):
            content_hash = hash(doc.page_content)
            keyword_score = doc.metadata.get('score', 0)
            
            if content_hash in doc_data:
                # Document found in both results - update keyword rank and score
                doc_data[content_hash]['keyword_rank'] = rank
                doc_data[content_hash]['keyword_score'] = keyword_score
            else:
                # Document only in keyword results
                doc_data[content_hash] = {
                    'document': doc,
                    'vector_rank': None,
                    'keyword_rank': rank,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score,
                    'content_hash': content_hash
                }
        
        # Calculate RRF scores and sort with detailed logging
        scored_docs = []
        rrf_calculation_start = time.time()
        
        for entry in doc_data.values():
            rrf_score = self._calculate_rrf_score(
                entry['vector_rank'], 
                entry['keyword_rank'],
                rrf_k
            )
            
            # Log RRF score calculation for debugging (first 3 documents)
            if len(scored_docs) < 3:
                found_by = []
                if entry['vector_rank'] is not None:
                    found_by.append("vector")
                if entry['keyword_rank'] is not None:
                    found_by.append("keyword")
                    
                logger.debug(f"RRF score calculation #{len(scored_docs) + 1}", extra={
                    "rrf_score": rrf_score,
                    "vector_rank": entry['vector_rank'],
                    "keyword_rank": entry['keyword_rank'],
                    "rrf_k": rrf_k,
                    "found_by_engines": found_by,
                    "content_preview": entry['document'].page_content[:100] + "..." if entry['document'].page_content else "[No content]"
                })
            
            # Determine which engines found this document
            found_by_engines = []
            if entry['vector_rank'] is not None:
                found_by_engines.append("vector")
            if entry['keyword_rank'] is not None:
                found_by_engines.append("keyword")
            
            # Update document metadata with comprehensive RRF information
            doc = entry['document']
            doc.metadata.update({
                'rrf_score': rrf_score,
                'vector_rank': entry['vector_rank'],
                'keyword_rank': entry['keyword_rank'],
                'vector_score': entry['vector_score'],
                'keyword_score': entry['keyword_score'],
                'found_by_engines': found_by_engines,
                'rrf_k_value': rrf_k,
                'fusion_method': 'reciprocal_rank_fusion'
            })
            
            scored_docs.append((rrf_score, doc))
        
        rrf_calculation_time = int((time.time() - rrf_calculation_start) * 1000)
        
        # Sort by RRF score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        fusion_time = int((time.time() - fusion_start) * 1000)
        
        # Log RRF fusion completion with analytics
        top_rrf_scores = [score for score, _ in scored_docs[:5]]  # Top 5 scores
        logger.debug("RRF fusion completed", extra={
            "total_fusion_time_ms": fusion_time,
            "rrf_calculation_time_ms": rrf_calculation_time,
            "unique_documents": len(scored_docs),
            "top_rrf_scores": top_rrf_scores,
            "rrf_k_used": rrf_k
        })
        
        return [doc for _, doc in scored_docs]
    
    def _calculate_rrf_score(
        self, 
        vector_rank: Optional[int], 
        keyword_rank: Optional[int],
        k: int = 60
    ) -> float:
        """Calculate Reciprocal Rank Fusion score using standard formula with detailed logging."""
        rrf_score = 0.0
        
        # Add vector contribution if document was found in vector results
        if vector_rank is not None:
            vector_contribution = 1.0 / (k + vector_rank)
            rrf_score += vector_contribution
        
        # Add keyword contribution if document was found in keyword results  
        if keyword_rank is not None:
            keyword_contribution = 1.0 / (k + keyword_rank)
            rrf_score += keyword_contribution
        
        return rrf_score
    
    def _calculate_rrf_metrics(self, documents: List[Document]) -> Dict[str, Any]:
        """Calculate RRF-specific metrics for monitoring and analysis."""
        if not documents:
            return {
                'engine_agreement_count': 0,
                'vector_only_count': 0,
                'keyword_only_count': 0,
                'avg_rrf_score': 0.0,
                'top_rrf_score': 0.0,
                'rrf_score_distribution': {}
            }
        
        engine_agreement_count = 0
        vector_only_count = 0
        keyword_only_count = 0
        rrf_scores = []
        
        for doc in documents:
            metadata = doc.metadata
            found_by = metadata.get('found_by_engines', [])
            rrf_score = metadata.get('rrf_score', 0.0)
            rrf_scores.append(rrf_score)
            
            if len(found_by) > 1:
                engine_agreement_count += 1
            elif 'vector' in found_by:
                vector_only_count += 1
            elif 'keyword' in found_by:
                keyword_only_count += 1
        
        # Calculate RRF score distribution
        if rrf_scores:
            avg_rrf = sum(rrf_scores) / len(rrf_scores)
            top_rrf = max(rrf_scores)
            
            # Group scores into buckets for distribution analysis
            score_buckets = {'high': 0, 'medium': 0, 'low': 0}
            for score in rrf_scores:
                if score >= 0.05:  # High RRF score (top-ranked in both engines)
                    score_buckets['high'] += 1
                elif score >= 0.01:  # Medium RRF score
                    score_buckets['medium'] += 1
                else:  # Low RRF score
                    score_buckets['low'] += 1
        else:
            avg_rrf = 0.0
            top_rrf = 0.0
            score_buckets = {'high': 0, 'medium': 0, 'low': 0}
        
        return {
            'engine_agreement_count': engine_agreement_count,
            'vector_only_count': vector_only_count,
            'keyword_only_count': keyword_only_count,
            'agreement_percentage': round((engine_agreement_count / len(documents) * 100), 2) if documents else 0,
            'avg_rrf_score': round(avg_rrf, 6),
            'top_rrf_score': round(top_rrf, 6),
            'rrf_score_distribution': score_buckets
        }


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