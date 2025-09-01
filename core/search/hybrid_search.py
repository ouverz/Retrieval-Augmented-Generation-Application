from __future__ import annotations
from typing import List
import pandas as pd
import logging
import time
from langchain.schema import Document

from core.search.bm25_search import BM25SearchEngine
from core.search.vector_search import VectorSearchEngine
from config.settings import HybridSearchConfig
from core.services.cache_service import get_cache_service
from core.services.synthesis_service import synthesize_answer

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

    def _create_rrf_hybrid_scores(self, bm25_docs: List[Document], vec_df: pd.DataFrame, query: str) -> List[Document]:
        """Create hybrid scores using Reciprocal Rank Fusion (RRF) algorithm with comprehensive logging."""
        rrf_start_time = time.time()
        
        # Log RRF fusion initiation with structured data
        logger.debug("Starting RRF fusion", extra={
            "bm25_docs_count": len(bm25_docs),
            "vector_docs_count": len(vec_df),
            "rrf_k_value": self.config.rrf_k,
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
            rrf_score = self._calculate_rrf_score(bm25_rank, vector_rank)
            
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
                    "rrf_k_value": self.config.rrf_k,
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
                "rrf_k_value": self.config.rrf_k,  # K parameter used for scoring
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
            "rrf_k_used": self.config.rrf_k
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

    async def search(self, query: str, top_k: int = None, remove_duplicates: bool = True, vector_weight: float = None):
        """Search using Reciprocal Rank Fusion (RRF) to combine BM25 and vector results.
        
        Note: vector_weight parameter is maintained for API compatibility but not used in RRF.
        RRF uses rank positions instead of weighted scores.
        """
        # Use top_k if provided, otherwise fall back to config max_results
        final_limit = top_k if top_k is not None else self.config.max_results
        
        # For RRF, we use the config directly (no weight calculations needed)
        cache_config = self.config
        
        # Check cache first
        cached_result = self.cache_service.get_cached_query_result(query, final_limit, cache_config)
        if cached_result is not None:
            ctx_df, response = cached_result
            logger.info("Cache hit for RRF hybrid search", extra={
                "query_preview": query[:50] + "..." if len(query) > 50 else query,
                "rrf_k_value": self.config.rrf_k,
                "cached_results_count": len(ctx_df) if ctx_df is not None else 0,
                "fusion_method": "reciprocal_rank_fusion"
            })
            return ctx_df, response
        
        # Cache miss - perform full search with structured logging
        logger.info("Cache miss - initiating RRF hybrid search", extra={
            "query_length": len(query),
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "final_limit": final_limit,
            "rrf_k_value": self.config.rrf_k,
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
        hybrid_results = self._create_rrf_hybrid_scores(bm25_docs, vec_df, query)
        
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
            "rrf_k_value": self.config.rrf_k,
            "fusion_method": "reciprocal_rank_fusion",
            "deduplication_applied": remove_duplicates
        })
        
        # Legacy log messages for compatibility
        logger.info(f"RRF Hybrid Search: {len(bm25_docs)} BM25 + {len(vec_df)} Vector → {len(hybrid_results)} unique → {len(top)} final")
        logger.info(f"Final results: {bm25_count} found by BM25, {vector_count} found by Vector, {both_count} found by both")
        logger.info(f"RRF parameters: k={self.config.rrf_k}")

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
