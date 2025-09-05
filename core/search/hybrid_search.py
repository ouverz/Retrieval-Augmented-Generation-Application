from __future__ import annotations
from typing import List, Dict, Optional, Any
import pandas as pd
import logging
import time
from langchain.schema import Document

from core.search.bm25_search import BM25SearchEngine
from core.search.vector_search import VectorSearchEngine
from config.settings import HybridSearchConfig
from core.services.cache_service import get_cache_service
from core.services.synthesis_service import synthesize_answer
from core.interfaces.search_engines import (
    SearchResult, VectorSearchEngine as IVectorSearchEngine, 
    KeywordSearchEngine as IKeywordSearchEngine,
    HybridSearchEngine as IHybridSearchEngine, SearchEngineFactory
)
from core.database.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search engine using Reciprocal Rank Fusion (RRF) to combine BM25 and Vector search results."""

    def __init__(
        self,
        bm25_engine: BM25SearchEngine,
        vector_engine: VectorSearchEngine,
        config: HybridSearchConfig,
    ) -> None:
        self.bm25_engine = bm25_engine
        self.vector_engine = vector_engine
        self.config = config
        self.cache_service = get_cache_service()

    def _score_bm25(self, docs: List[Document]) -> List[Document]:
        """Score BM25 documents with rank-based scoring for RRF."""
        for i, d in enumerate(docs):
            # Apply content quality penalty for citation-heavy chunks
            content = d.page_content or ""
            quality_penalty = self._calculate_content_quality_score(content)
            
            if not isinstance(d.metadata, dict):
                d.metadata = {}
            d.metadata.update({
                "source_engine": "bm25", 
                "rank": i + 1,
                "content_quality_penalty": quality_penalty
            })
        return docs
    
    def _calculate_content_quality_score(self, content: str) -> float:
        """Calculate content quality score (0.1 to 1.0) to penalize citation-heavy chunks"""
        import re
        
        if not content:
            return 0.1
        
        # Citation patterns (simplified from investigation script)
        citation_patterns = [
            r'\d{4}[.,;]',  # Years
            r'[A-Z][a-z]+\s+et\s+al[.,]',  # "Author et al."
            r'[A-Z][a-z]+,\s*[A-Z]\.',  # "Smith, J."
            r'pp?\.\s*\d+',  # Page numbers
            r'doi:', # DOI identifiers
            r'Published in final edited form as:', # Common citation text
        ]
        
        # Count citation matches
        word_count = len(content.split())
        citation_matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in citation_patterns)
        citation_ratio = citation_matches / max(word_count, 1)
        
        # Calculate quality score: high citation ratio = low quality score
        if citation_ratio > 0.15:  # Very citation heavy
            return 0.1
        elif citation_ratio > 0.1:  # Citation heavy
            return 0.3
        elif citation_ratio > 0.05:  # Some citations
            return 0.7
        else:  # Clean content
            return 1.0

    def _calculate_rrf_score(self, bm25_rank: int = None, vector_rank: int = None, k: int = None) -> float:
        """Calculate RRF score using Reciprocal Rank Fusion algorithm with detailed component tracking.
        
        RRF formula: score = sum(1 / (k + rank)) for each ranking
        
        Args:
            bm25_rank: Rank from BM25 search (1-indexed, None if not found)
            vector_rank: Rank from vector search (1-indexed, None if not found) 
            k: RRF constant (defaults to config.rrf_k)
        
        Returns:
            Combined RRF score
        """
        if k is None:
            k = self.config.rrf_k
            
        rrf_score = 0.0
        bm25_contribution = 0.0
        vector_contribution = 0.0
        
        if bm25_rank is not None:
            bm25_contribution = 1.0 / (k + bm25_rank)
            rrf_score += bm25_contribution
            
        if vector_rank is not None:
            vector_contribution = 1.0 / (k + vector_rank)
            rrf_score += vector_contribution
            
        return rrf_score

    def _create_rrf_hybrid_scores(self, bm25_docs: List[Document], vec_df: pd.DataFrame, query: str, rrf_k: int = None) -> List[Document]:
        """Create hybrid scores using Reciprocal Rank Fusion (RRF) algorithm with comprehensive logging."""
        rrf_start_time = time.time()
        
        # Use provided rrf_k or fall back to config
        effective_rrf_k = rrf_k if rrf_k is not None else self.config.rrf_k
        
        # Log RRF fusion initiation with structured data
        logger.debug("Starting RRF fusion", extra={
            "bm25_docs_count": len(bm25_docs),
            "vector_docs_count": len(vec_df),
            "rrf_k_value": effective_rrf_k,
            "query_preview": query[:50] + "..." if len(query) > 50 else query
        })
        
        # DEBUG: Log first few document details for debugging
        for i, doc in enumerate(bm25_docs[:3]):
            content_preview = (doc.page_content or "")[:100] + "..."
            doc_id = doc.metadata.get("id", "NO_ID")
            logger.debug(f"BM25 doc {i+1}", extra={
                "doc_id": doc_id,
                "rank": i+1,
                "content_preview": content_preview
            })
        
        for i, (_, row) in enumerate(vec_df.head(3).iterrows()):
            content_preview = (row["content"] or "")[:100] + "..."
            doc_id = row.get("id", "NO_ID")
            logger.debug(f"Vector doc {i+1}", extra={
                "doc_id": doc_id,
                "rank": i+1,
                "distance": row.get("distance", "N/A"),
                "content_preview": content_preview
            })
        
        # Build lookup dictionaries for efficient access
        bm25_lookup = {}  # content_hash -> (rank, doc)
        for i, doc in enumerate(bm25_docs):
            content_hash = hash((doc.page_content or "").strip().lower())
            bm25_lookup[content_hash] = (i + 1, doc)
            if i < 3:  # Debug first few hashes
                logger.debug(f"BM25 hash {i+1}: {content_hash}")
        
        vector_lookup = {}  # content_hash -> (similarity, row, rank)
        for i, (_, row) in enumerate(vec_df.iterrows()):
            content_hash = hash((row["content"] or "").strip().lower())
            distance = float(row["distance"])
            similarity = 1.0 / (1.0 + distance)
            vector_lookup[content_hash] = (similarity, row, i + 1)
            if i < 3:  # Debug first few hashes
                logger.debug(f"Vector hash {i+1}: {content_hash}")
        
        # Check for hash overlaps (document-level fusion opportunities)
        hash_overlaps = set(bm25_lookup.keys()) & set(vector_lookup.keys())
        logger.debug(f"Found {len(hash_overlaps)} content hash overlaps (fusion opportunities)")
        
        # Collect all unique documents
        all_content_hashes = set(bm25_lookup.keys()) | set(vector_lookup.keys())
        
        hybrid_results = []
        for content_hash in all_content_hashes:
            # Initialize variables
            bm25_rank = None
            vector_rank = None
            quality_penalty = 1.0
            found_by = []
            vector_similarity = None
            vector_distance = None
            
            # Get BM25 information
            if content_hash in bm25_lookup:
                rank, doc = bm25_lookup[content_hash]
                content = doc.page_content or ""
                quality_penalty = self._calculate_content_quality_score(content)
                bm25_rank = rank
                found_by.append("bm25")
                base_doc = doc
            
            # Get Vector information
            if content_hash in vector_lookup:
                similarity, row, rank = vector_lookup[content_hash]
                vector_similarity = similarity
                vector_distance = float(row["distance"])
                vector_rank = rank
                found_by.append("vector")
                
                # Use vector document if not found by BM25
                if content_hash not in bm25_lookup:
                    from langchain.schema import Document
                    base_doc = Document(
                        page_content=row["content"],
                        metadata={"id": row["id"]}
                    )
            
            # Calculate RRF score with detailed component tracking
            rrf_score = self._calculate_rrf_score(bm25_rank, vector_rank, k=effective_rrf_k)
            
            # Apply quality penalty to final score
            final_score = rrf_score * quality_penalty
            
            # Log RRF scoring details for first few documents for debugging
            if len(hybrid_results) < 3:
                logger.debug(f"RRF scoring #{len(hybrid_results) + 1}", extra={
                    "raw_rrf_score": round(rrf_score, 6),
                    "final_rrf_score": round(final_score, 6),
                    "bm25_rank": bm25_rank,
                    "vector_rank": vector_rank,
                    "quality_penalty": round(quality_penalty, 3),
                    "found_by_engines": found_by,
                    "rrf_k_value": effective_rrf_k,
                    "content_preview": (base_doc.page_content or "")[:100] + "..." if base_doc.page_content else "[No content]"
                })
            
            # Create enhanced metadata with RRF scoring information per requirements
            base_doc.metadata.update({
                "rrf_score": final_score,  # Final RRF score
                "raw_rrf_score": rrf_score,  # Raw RRF score before quality adjustment
                "bm25_rank": bm25_rank,  # Position in BM25 results (if found)
                "vector_rank": vector_rank,  # Position in Vector results (if found)
                "vector_similarity": vector_similarity,
                "vector_distance": vector_distance,
                "content_quality_penalty": quality_penalty,
                "found_by_engines": found_by,  # List of engines that found this document
                "rrf_k_value": effective_rrf_k,  # K parameter used for scoring
                "source_engine": "rrf_hybrid",  # Mark as RRF hybrid result
                "fusion_method": "reciprocal_rank_fusion"
            })
            
            hybrid_results.append(base_doc)
        
        # Calculate and log RRF fusion completion metrics
        rrf_time_ms = int((time.time() - rrf_start_time) * 1000)
        
        # Calculate agreement and ranking statistics
        both_engines = len([r for r in hybrid_results if len(r.metadata.get("found_by_engines", [])) > 1])
        bm25_only = len([r for r in hybrid_results if r.metadata.get("found_by_engines", []) == ["bm25"]])
        vector_only = len([r for r in hybrid_results if r.metadata.get("found_by_engines", []) == ["vector"]])
        
        rrf_scores = [r.metadata.get("rrf_score", 0.0) for r in hybrid_results]
        avg_rrf_score = sum(rrf_scores) / len(rrf_scores) if rrf_scores else 0.0
        top_rrf_score = max(rrf_scores) if rrf_scores else 0.0
        
        logger.debug("RRF fusion completed", extra={
            "rrf_processing_time_ms": rrf_time_ms,
            "total_unique_documents": len(hybrid_results),
            "both_engines_found": both_engines,
            "bm25_only_found": bm25_only,
            "vector_only_found": vector_only,
            "agreement_percentage": round((both_engines / len(hybrid_results) * 100), 2) if hybrid_results else 0,
            "avg_rrf_score": round(avg_rrf_score, 6),
            "top_rrf_score": round(top_rrf_score, 6),
            "rrf_k_used": effective_rrf_k
        })
        
        return hybrid_results

    def _score_vector(self, df: pd.DataFrame) -> List[Document]:
        """Score vector documents with rank-based scoring for RRF."""
        req = {"id", "content", "distance"}
        if not req.issubset(df.columns):
            raise ValueError(f"Vector results missing columns: {req - set(df.columns)}")
        out: List[Document] = []
        for i, row in df.iterrows():
            sim = 1.0 / (1.0 + float(row["distance"]))
            meta = {
                "source_engine": "vector",
                "id": row["id"],
                "distance": float(row["distance"]),
                "similarity": sim,
                "rank": i + 1,
            }
            out.append(Document(page_content=row["content"], metadata=meta))
        return out

    def _dedupe(self, docs: List[Document]) -> List[Document]:
        uniq, seen = [], set()
        for d in docs:
            key = hash((d.page_content or "").strip().lower())
            if key not in seen:
                seen.add(key)
                uniq.append(d)
        return uniq

    def _to_df(self, docs: List[Document]) -> pd.DataFrame:
        rows = []
        for d in docs:
            m = d.metadata if isinstance(d.metadata, dict) else {}
            # Fix: Use source_engine to generate appropriate fallback ID
            if m.get("id"):
                rid = m.get("id")
            else:
                engine = m.get("source_engine", "unknown")
                source = m.get("source", "")
                chunk_id = m.get("chunk_id", "")
                rid = f"{engine}:{source}-{chunk_id}"
            
            rows.append(
                {"id": str(rid), "content": d.page_content or "", "metadata": m}
            )
        return pd.DataFrame(rows)

    async def search(self, query: str, top_k: int = None, remove_duplicates: bool = True, vector_weight: float = None, rrf_k: int = None):
        """Search using Reciprocal Rank Fusion (RRF) to combine BM25 and vector results.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            remove_duplicates: Whether to deduplicate results
            vector_weight: Maintained for API compatibility but not used in RRF
            rrf_k: RRF k parameter override (uses config.rrf_k if None)
        
        Note: vector_weight parameter is maintained for API compatibility but not used in RRF.
        RRF uses rank positions instead of weighted scores.
        """
        # Use top_k if provided, otherwise fall back to config max_results
        final_limit = top_k if top_k is not None else self.config.max_results
        
        # Use provided rrf_k or fall back to config
        effective_rrf_k = rrf_k if rrf_k is not None else self.config.rrf_k
        
        # For RRF, we use the config directly (no weight calculations needed)
        cache_config = self.config
        
        # Check cache first
        cached_result = self.cache_service.get_cached_query_result(query, final_limit, cache_config)
        if cached_result is not None:
            ctx_df, response = cached_result
            logger.info("Cache hit for RRF hybrid search", extra={
                "query_preview": query[:50] + "..." if len(query) > 50 else query,
                "rrf_k_value": effective_rrf_k,
                "cached_results_count": len(ctx_df) if ctx_df is not None else 0,
                "fusion_method": "reciprocal_rank_fusion"
            })
            return ctx_df, response
        
        # Cache miss - perform full search with structured logging
        logger.info("Cache miss - initiating RRF hybrid search", extra={
            "query_length": len(query),
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "final_limit": final_limit,
            "rrf_k_value": effective_rrf_k,
            "bm25_top_k": self.config.bm25_top_k,
            "vector_top_k": self.config.vector_top_k,
            "deduplication_enabled": remove_duplicates,
            "fusion_method": "reciprocal_rank_fusion"
        })
        
        # Get results from both engines
        bm25_docs = self.bm25_engine.search(query, self.config.bm25_top_k)
        vec_df = self.vector_engine.search(
            query, self.config.vector_top_k, return_dataframe=True
        )
        
        # Create RRF hybrid scoring 
        hybrid_results = self._create_rrf_hybrid_scores(bm25_docs, vec_df, query, rrf_k=effective_rrf_k)
        
        # Apply deduplication if requested
        if remove_duplicates:
            hybrid_results = self._dedupe(hybrid_results)
        
        # Sort by RRF score
        hybrid_results.sort(key=lambda d: d.metadata.get("rrf_score", 0.0), reverse=True)
        
        top = hybrid_results[:final_limit]
        ctx_df = self._to_df(top)
        
        # Calculate comprehensive RRF hybrid search metrics
        bm25_count = len([r for r in top if "bm25" in r.metadata.get("found_by_engines", [])])
        vector_count = len([r for r in top if "vector" in r.metadata.get("found_by_engines", [])])
        both_count = len([r for r in top if len(r.metadata.get("found_by_engines", [])) > 1])
        
        # Calculate RRF score distribution for monitoring
        rrf_scores = [r.metadata.get("rrf_score", 0.0) for r in top]
        avg_rrf_score = sum(rrf_scores) / len(rrf_scores) if rrf_scores else 0.0
        top_rrf_score = max(rrf_scores) if rrf_scores else 0.0
        
        # Log comprehensive RRF hybrid search completion with structured data
        logger.info("RRF hybrid search completed", extra={
            "input_bm25_results": len(bm25_docs),
            "input_vector_results": len(vec_df),
            "unique_results_after_fusion": len(hybrid_results),
            "final_results_count": len(top),
            "bm25_only_count": bm25_count - both_count,
            "vector_only_count": vector_count - both_count,
            "both_engines_count": both_count,
            "engine_agreement_rate": round((both_count / len(top) * 100), 2) if top else 0,
            "avg_rrf_score": round(avg_rrf_score, 6),
            "top_rrf_score": round(top_rrf_score, 6),
            "rrf_k_value": effective_rrf_k,
            "fusion_method": "reciprocal_rank_fusion",
            "deduplication_applied": remove_duplicates
        })
        
        # Legacy log messages for compatibility
        logger.info(f"RRF Hybrid Search: {len(bm25_docs)} BM25 + {len(vec_df)} Vector → {len(hybrid_results)} unique → {len(top)} final")
        logger.info(f"Final results: {bm25_count} found by BM25, {vector_count} found by Vector, {both_count} found by both")
        logger.info(f"RRF parameters: k={effective_rrf_k}")

        # Generate response
        try:
            response = await synthesize_answer(query=query, context=ctx_df)
            
            # Cache the complete result
            self.cache_service.cache_query_result(query, final_limit, cache_config, ctx_df, response)
            
            return ctx_df, response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Return ctx_df and None so the caller can handle the synthesis
            return ctx_df, None


# Clean interface implementations for better separation of concerns

class CleanVectorSearchEngine(IVectorSearchEngine):
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
        return self.vector_store is not None


class CleanKeywordSearchEngine(IKeywordSearchEngine):
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


class CleanHybridSearchEngine(IHybridSearchEngine):
    """Clean hybrid search engine combining vector and keyword search."""
    
    def __init__(
        self, 
        vector_engine: IVectorSearchEngine,
        keyword_engine: IKeywordSearchEngine,
        default_vector_weight: float = 0.6
    ):
        self.vector_engine = vector_engine
        self.keyword_engine = keyword_engine
        self.default_vector_weight = default_vector_weight
        
    def search(self, query: str, top_k: int = 10, 
              vector_weight: Optional[float] = None,
              rrf_k: Optional[int] = None) -> SearchResult:
        """Perform hybrid search using Reciprocal Rank Fusion (RRF)."""
        start_time = time.time()
        effective_rrf_k = rrf_k if rrf_k is not None else 60
        
        logger.info("RRF hybrid search initiated", extra={
            "query_length": len(query),
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "top_k": top_k,
            "rrf_k_value": effective_rrf_k,
            "search_type": "hybrid_rrf",
            "fusion_method": "reciprocal_rank_fusion"
        })
        
        try:
            # Perform both searches
            vector_result = self.vector_engine.search(query, top_k=top_k)
            keyword_result = self.keyword_engine.search(query, top_k=top_k)
            
            # Combine results using RRF
            combined_docs = self._document_level_fusion(
                vector_result.documents,
                keyword_result.documents,
                effective_rrf_k,
                query
            )
            
            # Limit to top_k results
            combined_docs = combined_docs[:top_k]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Calculate RRF metrics
            rrf_metrics = self._calculate_rrf_metrics(combined_docs)
            
            enhanced_metadata = {
                'search_type': 'hybrid_rrf',
                'fusion_method': 'reciprocal_rank_fusion',
                'rrf_k_value': effective_rrf_k,
                'vector_results': len(vector_result.documents),
                'keyword_results': len(keyword_result.documents),
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
        rrf_k: int = 60,
        query: str = ""
    ) -> List[Document]:
        """Perform RRF-based fusion of search results."""
        doc_data = {}
        
        # Process vector results
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
        
        # Process keyword results
        for rank, doc in enumerate(keyword_docs, 1):
            content_hash = hash(doc.page_content)
            keyword_score = doc.metadata.get('score', 0)
            
            if content_hash in doc_data:
                doc_data[content_hash]['keyword_rank'] = rank
                doc_data[content_hash]['keyword_score'] = keyword_score
            else:
                doc_data[content_hash] = {
                    'document': doc,
                    'vector_rank': None,
                    'keyword_rank': rank,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score,
                    'content_hash': content_hash
                }
        
        # Calculate RRF scores and sort
        scored_docs = []
        for entry in doc_data.values():
            rrf_score = self._calculate_rrf_score(
                entry['vector_rank'], 
                entry['keyword_rank'],
                rrf_k
            )
            
            # Determine which engines found this document
            found_by_engines = []
            if entry['vector_rank'] is not None:
                found_by_engines.append("vector")
            if entry['keyword_rank'] is not None:
                found_by_engines.append("keyword")
            
            # Update document metadata
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
        
        # Sort by RRF score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs]
    
    def _calculate_rrf_score(
        self, 
        vector_rank: Optional[int], 
        keyword_rank: Optional[int],
        k: int = 60
    ) -> float:
        """Calculate Reciprocal Rank Fusion score."""
        rrf_score = 0.0
        
        if vector_rank is not None:
            rrf_score += 1.0 / (k + vector_rank)
        
        if keyword_rank is not None:
            rrf_score += 1.0 / (k + keyword_rank)
        
        return rrf_score
    
    def _calculate_rrf_metrics(self, documents: List[Document]) -> Dict[str, Any]:
        """Calculate RRF-specific metrics for monitoring."""
        if not documents:
            return {
                'engine_agreement_count': 0,
                'vector_only_count': 0,
                'keyword_only_count': 0,
                'avg_rrf_score': 0.0,
                'top_rrf_score': 0.0
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
        
        avg_rrf = sum(rrf_scores) / len(rrf_scores) if rrf_scores else 0.0
        top_rrf = max(rrf_scores) if rrf_scores else 0.0
        
        return {
            'engine_agreement_count': engine_agreement_count,
            'vector_only_count': vector_only_count,
            'keyword_only_count': keyword_only_count,
            'agreement_percentage': round((engine_agreement_count / len(documents) * 100), 2) if documents else 0,
            'avg_rrf_score': round(avg_rrf, 6),
            'top_rrf_score': round(top_rrf, 6)
        }


class CleanSearchEngineFactory(SearchEngineFactory):
    """Factory for creating clean search engine instances."""
    
    def __init__(self, vector_store: VectorStore, bm25_engine: BM25SearchEngine):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine
        
    def create_vector_engine(self) -> IVectorSearchEngine:
        """Create vector search engine instance."""
        return CleanVectorSearchEngine(self.vector_store)
    
    def create_keyword_engine(self) -> IKeywordSearchEngine:
        """Create keyword search engine instance."""
        return CleanKeywordSearchEngine(self.bm25_engine)
    
    def create_hybrid_engine(
        self, 
        vector_engine: IVectorSearchEngine,
        keyword_engine: IKeywordSearchEngine,
        vector_weight: float = 0.6
    ) -> IHybridSearchEngine:
        """Create hybrid search engine instance."""
        return CleanHybridSearchEngine(
            vector_engine, 
            keyword_engine, 
            vector_weight
        )
