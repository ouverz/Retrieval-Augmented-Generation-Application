"""Reciprocal Rank Fusion (RRF) algorithm implementation.

This module provides a standalone implementation of the Reciprocal Rank Fusion
algorithm for combining ranked lists from multiple search engines.

RRF Formula: score = 1 / (k + rank)
where k is a constant (default 60) and rank is the position in the ranked list.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_rrf_scores(
    ranked_results: List[List[Tuple[str, int]]], 
    k: int = 60
) -> Dict[str, float]:
    """Calculate RRF scores for documents from multiple ranked lists.
    
    Args:
        ranked_results: List of ranked lists, where each inner list contains
                       (doc_id, rank) tuples from a search engine
        k: RRF constant parameter (default 60, as commonly used)
    
    Returns:
        Dictionary mapping doc_id to combined RRF score
        
    Raises:
        ValueError: If k is negative or ranked_results is empty
    """
    if k < 0:
        raise ValueError("k parameter must be non-negative")
    
    if not ranked_results:
        raise ValueError("ranked_results cannot be empty")
    
    # Remove empty lists and log them
    non_empty_results = [results for results in ranked_results if results]
    if len(non_empty_results) != len(ranked_results):
        empty_count = len(ranked_results) - len(non_empty_results)
        logger.warning(f"Skipped {empty_count} empty result lists")
    
    if not non_empty_results:
        logger.warning("All result lists are empty, returning empty scores")
        return {}
    
    # Calculate RRF scores
    rrf_scores: Dict[str, float] = {}
    
    for engine_idx, results in enumerate(non_empty_results):
        logger.debug(f"Processing engine {engine_idx + 1} with {len(results)} results")
        
        for doc_id, rank in results:
            if rank <= 0:
                logger.warning(f"Invalid rank {rank} for doc {doc_id}, skipping")
                continue
                
            # RRF formula: score = 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)
            
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += rrf_score
            else:
                rrf_scores[doc_id] = rrf_score
    
    logger.debug(f"RRF fusion computed scores for {len(rrf_scores)} unique documents")
    return rrf_scores


def fuse_search_results(
    ranked_results: List[List[Tuple[str, int]]],
    k: int = 60,
    top_k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists using RRF and return sorted results.
    
    Args:
        ranked_results: List of ranked lists from different search engines
        k: RRF constant parameter (default 60)
        top_k: Maximum number of results to return (None for all)
    
    Returns:
        List of (doc_id, rrf_score) tuples sorted by RRF score (descending)
        
    Raises:
        ValueError: If k is negative or ranked_results is empty
    """
    rrf_scores = calculate_rrf_scores(ranked_results, k)
    
    # Sort by RRF score descending
    sorted_results = sorted(
        rrf_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Apply top_k limit if specified
    if top_k is not None:
        if top_k <= 0:
            return []
        sorted_results = sorted_results[:top_k]
    
    return sorted_results


class RRFFusion:
    """Reciprocal Rank Fusion implementation with configurable parameters."""
    
    def __init__(self, k: int = 60):
        """Initialize RRF fusion with specified k parameter.
        
        Args:
            k: RRF constant parameter (default 60)
            
        Raises:
            ValueError: If k is negative
        """
        if k < 0:
            raise ValueError("k parameter must be non-negative")
        self.k = k
        logger.info(f"Initialized RRF fusion with k={k}")
    
    def calculate_scores(
        self, 
        ranked_results: List[List[Tuple[str, int]]]
    ) -> Dict[str, float]:
        """Calculate RRF scores using the configured k parameter."""
        return calculate_rrf_scores(ranked_results, self.k)
    
    def fuse_results(
        self,
        ranked_results: List[List[Tuple[str, int]]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Fuse results using the configured k parameter."""
        return fuse_search_results(ranked_results, self.k, top_k)