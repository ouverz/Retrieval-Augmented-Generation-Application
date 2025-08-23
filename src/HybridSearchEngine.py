from __future__ import annotations
from typing import List
import pandas as pd
from langchain.schema import Document

from src.BM25SearchEngine import BM25SearchEngine
from src.VectorSearchEngine import VectorSearchEngine
from src.config.settings import HybridSearchConfig
from src.services.cache_service import get_cache_service
from src.services.synthesizer import synthesize_answer


class HybridSearchEngine:
    """Combine BM25 + Vector scores, dedupe, rank, synthesize."""

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
        for i, d in enumerate(docs):
            base_score = self.config.bm25_weight * (1.0 / (i + 1))
            
            # Apply content quality penalty for citation-heavy chunks
            content = d.page_content or ""
            quality_penalty = self._calculate_content_quality_score(content)
            
            # Final score with quality adjustment
            s = base_score * quality_penalty
            
            if not isinstance(d.metadata, dict):
                d.metadata = {}
            d.metadata.update({
                "score": s, 
                "source_engine": "bm25", 
                "rank": i + 1,
                "base_bm25_score": base_score,
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

    def _create_true_hybrid_scores(self, bm25_docs: List[Document], vec_df: pd.DataFrame, query: str) -> List[Document]:
        """Create true hybrid scores where each document gets both BM25 and Vector scores"""
        
        # DEBUG: Log document details to identify matching issues
        print(f"DEBUG: BM25 docs ({len(bm25_docs)}):")
        for i, doc in enumerate(bm25_docs[:3]):  # Show first 3
            content_preview = (doc.page_content or "")[:100] + "..."
            doc_id = doc.metadata.get("id", "NO_ID")
            print(f"  {i+1}. ID: {doc_id}, Content: {content_preview}")
        
        print(f"DEBUG: Vector docs ({len(vec_df)}):")
        for i, (_, row) in enumerate(vec_df.head(3).iterrows()):  # Show first 3
            content_preview = (row["content"] or "")[:100] + "..."
            doc_id = row.get("id", "NO_ID")
            print(f"  {i+1}. ID: {doc_id}, Content: {content_preview}")
        
        # Build lookup dictionaries for efficient access
        bm25_lookup = {}  # content_hash -> (rank, doc)
        for i, doc in enumerate(bm25_docs):
            content_hash = hash((doc.page_content or "").strip().lower())
            bm25_lookup[content_hash] = (i + 1, doc)
            if i < 3:  # Debug first few hashes
                print(f"DEBUG: BM25 hash {i+1}: {content_hash}")
        
        vector_lookup = {}  # content_hash -> (similarity, row)
        for i, (_, row) in enumerate(vec_df.iterrows()):
            content_hash = hash((row["content"] or "").strip().lower())
            distance = float(row["distance"])
            similarity = 1.0 / (1.0 + distance)
            vector_lookup[content_hash] = (similarity, row, i + 1)
            if i < 3:  # Debug first few hashes
                print(f"DEBUG: Vector hash {i+1}: {content_hash}")
        
        # Collect all unique documents
        all_content_hashes = set(bm25_lookup.keys()) | set(vector_lookup.keys())
        
        hybrid_results = []
        for content_hash in all_content_hashes:
            # Get BM25 score
            bm25_score = 0.0
            bm25_rank = None
            quality_penalty = 1.0
            found_by = []
            
            if content_hash in bm25_lookup:
                rank, doc = bm25_lookup[content_hash]
                content = doc.page_content or ""
                quality_penalty = self._calculate_content_quality_score(content)
                bm25_raw_score = 1.0 / rank
                bm25_score = self.config.bm25_weight * bm25_raw_score * quality_penalty
                bm25_rank = rank
                found_by.append("bm25")
                base_doc = doc
            
            # Get Vector score
            vector_score = 0.0
            vector_similarity = None
            vector_distance = None
            vector_rank = None
            
            if content_hash in vector_lookup:
                similarity, row, rank = vector_lookup[content_hash]
                vector_similarity = similarity
                vector_distance = float(row["distance"])
                vector_score = self.config.vector_weight * vector_distance  # Use distance for apples-to-apples comparison with BM25
                vector_rank = rank
                found_by.append("vector")
                
                # Use vector document if not found by BM25
                if content_hash not in bm25_lookup:
                    from langchain.schema import Document
                    base_doc = Document(
                        page_content=row["content"],
                        metadata={"id": row["id"]}
                    )
            
            # Calculate true hybrid score
            hybrid_score = bm25_score + vector_score
            
            # Create enhanced metadata
            base_doc.metadata.update({
                "hybrid_score": hybrid_score,
                "bm25_score": bm25_score,
                "vector_score": vector_score,
                "bm25_rank": bm25_rank,
                "vector_rank": vector_rank,
                "vector_similarity": vector_similarity,
                "vector_distance": vector_distance,
                "content_quality_penalty": quality_penalty,
                "found_by_engines": found_by,
                "source_engine": "hybrid"  # Mark as true hybrid result
            })
            
            hybrid_results.append(base_doc)
        
        return hybrid_results

    def _score_vector(self, df: pd.DataFrame) -> List[Document]:
        req = {"id", "content", "distance"}
        if not req.issubset(df.columns):
            raise ValueError(f"Vector results missing columns: {req - set(df.columns)}")
        out: List[Document] = []
        for i, row in df.iterrows():
            sim = 1.0 / (1.0 + float(row["distance"]))
            s = self.config.vector_weight * float(row["distance"])  # Use distance for apples-to-apples comparison with BM25
            meta = {
                "score": s,
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

    async def search(self, query: str, top_k: int = None, remove_duplicates: bool = True):
        # Use top_k if provided, otherwise fall back to config max_results
        final_limit = top_k if top_k is not None else self.config.max_results
        
        # Check cache first
        cached_result = self.cache_service.get_cached_query_result(query, final_limit, self.config)
        if cached_result is not None:
            ctx_df, response = cached_result
            print(f"[Backend] Cache hit for query: {query[:50]}...")
            return ctx_df, response
        
        # Cache miss - perform full search
        print(f"[Backend] Cache miss - performing full hybrid search")
        
        # Get results from both engines
        bm25_docs = self.bm25_engine.search(query, self.config.bm25_top_k)
        vec_df = self.vector_engine.search(
            query, self.config.vector_top_k, return_dataframe=True
        )
        
        # Create true hybrid scoring
        hybrid_results = self._create_true_hybrid_scores(bm25_docs, vec_df, query)
        
        # Apply deduplication if requested
        if remove_duplicates:
            hybrid_results = self._dedupe(hybrid_results)
        
        # Sort by true hybrid score
        hybrid_results.sort(key=lambda d: d.metadata.get("hybrid_score", 0.0), reverse=True)
        
        top = hybrid_results[:final_limit]
        ctx_df = self._to_df(top)
        
        # Log hybrid search summary
        bm25_count = len([r for r in top if "bm25" in r.metadata.get("found_by_engines", [])])
        vector_count = len([r for r in top if "vector" in r.metadata.get("found_by_engines", [])])
        both_count = len([r for r in top if len(r.metadata.get("found_by_engines", [])) > 1])
        
        print(f"True Hybrid Search: {len(bm25_docs)} BM25 + {len(vec_df)} Vector → {len(hybrid_results)} unique → {len(top)} final")
        print(f"Final results: {bm25_count} found by BM25, {vector_count} found by Vector, {both_count} found by both")

        # Generate response
        try:
            response = await synthesize_answer(query=query, context=ctx_df)
            
            # Cache the complete result
            self.cache_service.cache_query_result(query, final_limit, self.config, ctx_df, response)
            
            return ctx_df, response
        except Exception as e:
            print(f"[Backend] Failed to generate response: {e}")
            # Return ctx_df and None so the caller can handle the synthesis
            return ctx_df, None
