#!/usr/bin/env python3
"""
Proposed fix for true hybrid scoring implementation.
This shows the architectural changes needed.
"""

# CURRENT BROKEN APPROACH:
def current_broken_approach(self, query: str):
    """Current implementation - result fusion, not score fusion"""
    
    # Each engine scores its own results
    bm25_docs = self.bm25_engine.search(query, 10)  # Get top 10 from BM25
    bm25_scored = self._score_bm25(bm25_docs)        # Score with BM25 only
    
    vec_df = self.vector_engine.search(query, 10)    # Get top 10 from Vector
    vec_scored = self._score_vector(vec_df)          # Score with Vector only
    
    # Combine separate result sets
    combined = bm25_scored + vec_scored              # Just concatenate lists
    combined.sort(key=lambda d: d.metadata.get("score", 0.0), reverse=True)
    
    # Result: Each doc has EITHER BM25 OR Vector score, not both


# PROPOSED TRUE HYBRID APPROACH:
def true_hybrid_approach(self, query: str):
    """True hybrid scoring - each document scored by BOTH engines"""
    
    # Step 1: Get larger candidate pools from both engines
    bm25_docs = self.bm25_engine.search(query, 20)   # Larger pool
    vec_df = self.vector_engine.search(query, 20)    # Larger pool
    
    # Step 2: Create unified document pool (deduped)
    all_docs = self._create_unified_document_pool(bm25_docs, vec_df)
    
    # Step 3: Score EACH document with BOTH engines
    hybrid_results = []
    for doc in all_docs:
        bm25_score = self._get_bm25_score_for_doc(doc, query)
        vector_score = self._get_vector_score_for_doc(doc, query)
        
        # True hybrid score
        hybrid_score = (self.config.bm25_weight * bm25_score + 
                       self.config.vector_weight * vector_score)
        
        doc.metadata.update({
            "bm25_score": bm25_score,
            "vector_score": vector_score, 
            "hybrid_score": hybrid_score,
            "source_engines": "both"  # Scored by both engines
        })
        hybrid_results.append(doc)
    
    # Step 4: Rank by true hybrid scores
    hybrid_results.sort(key=lambda d: d.metadata["hybrid_score"], reverse=True)
    
    # Result: Each doc has BM25 score, Vector score, AND true hybrid score


# IMPLEMENTATION CHALLENGES:
"""
1. BM25 scoring is rank-based (1/rank) - need document-specific BM25 scores
2. Vector scoring needs document embeddings - may require re-embedding
3. Performance: Scoring each doc with both engines is expensive
4. Document matching: Same document may have different IDs in each engine
"""

# PRACTICAL COMPROMISE SOLUTION:
def practical_hybrid_approach(self, query: str):
    """Practical fix: Normalize scores and create weighted combinations"""
    
    # Get results from both engines
    bm25_docs = self.bm25_engine.search(query, 15)
    vec_df = self.vector_engine.search(query, 15)
    
    # Convert to unified format with normalized scores
    bm25_results = self._normalize_bm25_results(bm25_docs, query)
    vector_results = self._normalize_vector_results(vec_df, query)
    
    # Combine and create weighted scores
    all_results = {}  # doc_id -> result info
    
    # Add BM25 results
    for doc in bm25_results:
        doc_id = self._get_doc_id(doc)
        all_results[doc_id] = {
            "document": doc,
            "bm25_score": doc.metadata["normalized_bm25_score"],
            "vector_score": 0.0,  # Default if not found by vector search
            "found_by": ["bm25"]
        }
    
    # Add/update with Vector results
    for doc in vector_results:
        doc_id = self._get_doc_id(doc)
        if doc_id in all_results:
            # Found by both engines
            all_results[doc_id]["vector_score"] = doc.metadata["normalized_vector_score"]
            all_results[doc_id]["found_by"].append("vector")
        else:
            # Found only by vector search
            all_results[doc_id] = {
                "document": doc,
                "bm25_score": 0.0,  # Default if not found by BM25
                "vector_score": doc.metadata["normalized_vector_score"],
                "found_by": ["vector"]
            }
    
    # Calculate true hybrid scores
    final_results = []
    for doc_id, info in all_results.items():
        hybrid_score = (self.config.bm25_weight * info["bm25_score"] + 
                       self.config.vector_weight * info["vector_score"])
        
        doc = info["document"]
        doc.metadata.update({
            "bm25_score": info["bm25_score"],
            "vector_score": info["vector_score"],
            "hybrid_score": hybrid_score,
            "found_by_engines": info["found_by"]
        })
        final_results.append(doc)
    
    # Sort by true hybrid score
    final_results.sort(key=lambda d: d.metadata["hybrid_score"], reverse=True)
    
    return final_results[:self.config.max_results]

print("This file shows the architectural fix needed for true hybrid scoring")