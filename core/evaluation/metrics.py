"""
Information Retrieval evaluation metrics for RAG system assessment.
Provides standard IR metrics like MRR, MAP, Precision@K, Recall@K, and NDCG@K.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metric results."""
    value: float
    confidence_interval: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class EvaluationMetric(ABC):
    """Base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, 
                retrieved_docs: List[str], 
                relevant_docs: Set[str],
                scores: Optional[List[float]] = None) -> EvaluationResult:
        """
        Compute the metric for a single query.
        
        Args:
            retrieved_docs: List of retrieved document IDs in ranking order
            relevant_docs: Set of relevant document IDs for the query
            scores: Optional relevance scores for graded evaluation
            
        Returns:
            EvaluationResult with computed metric value
        """
        pass


class MeanReciprocalRank(EvaluationMetric):
    """Mean Reciprocal Rank (MRR) metric."""
    
    def compute(self, 
                retrieved_docs: List[str], 
                relevant_docs: Set[str],
                scores: Optional[List[float]] = None) -> EvaluationResult:
        """Compute MRR for a single query."""
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                rr_value = 1.0 / rank
                return EvaluationResult(
                    value=rr_value,
                    metadata={"first_relevant_rank": rank}
                )
        
        return EvaluationResult(value=0.0, metadata={"first_relevant_rank": None})


class PrecisionAtK(EvaluationMetric):
    """Precision@K metric."""
    
    def __init__(self, k: int):
        """Initialize with specific K value."""
        self.k = k
    
    def compute(self, 
                retrieved_docs: List[str], 
                relevant_docs: Set[str],
                scores: Optional[List[float]] = None) -> EvaluationResult:
        """Compute Precision@K for a single query."""
        top_k_docs = retrieved_docs[:self.k]
        relevant_count = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
        precision = relevant_count / len(top_k_docs) if top_k_docs else 0.0
        
        return EvaluationResult(
            value=precision,
            metadata={
                "k": self.k,
                "relevant_in_top_k": relevant_count,
                "total_retrieved": len(top_k_docs)
            }
        )


class RecallAtK(EvaluationMetric):
    """Recall@K metric."""
    
    def __init__(self, k: int):
        """Initialize with specific K value."""
        self.k = k
    
    def compute(self, 
                retrieved_docs: List[str], 
                relevant_docs: Set[str],
                scores: Optional[List[float]] = None) -> EvaluationResult:
        """Compute Recall@K for a single query."""
        if not relevant_docs:
            return EvaluationResult(value=0.0, metadata={"k": self.k, "no_relevant_docs": True})
            
        top_k_docs = retrieved_docs[:self.k]
        relevant_count = sum(1 for doc_id in top_k_docs if doc_id in relevant_docs)
        recall = relevant_count / len(relevant_docs)
        
        return EvaluationResult(
            value=recall,
            metadata={
                "k": self.k,
                "relevant_in_top_k": relevant_count,
                "total_relevant": len(relevant_docs)
            }
        )


class MeanAveragePrecision(EvaluationMetric):
    """Mean Average Precision (MAP) metric."""
    
    def compute(self, 
                retrieved_docs: List[str], 
                relevant_docs: Set[str],
                scores: Optional[List[float]] = None) -> EvaluationResult:
        """Compute Average Precision for a single query."""
        if not relevant_docs:
            return EvaluationResult(value=0.0, metadata={"no_relevant_docs": True})
        
        precision_sum = 0.0
        relevant_count = 0
        
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_sum += precision_at_rank
        
        avg_precision = precision_sum / len(relevant_docs) if relevant_docs else 0.0
        
        return EvaluationResult(
            value=avg_precision,
            metadata={
                "relevant_found": relevant_count,
                "total_relevant": len(relevant_docs),
                "precision_sum": precision_sum
            }
        )


class NDCGAtK(EvaluationMetric):
    """Normalized Discounted Cumulative Gain@K metric."""
    
    def __init__(self, k: int):
        """Initialize with specific K value."""
        self.k = k
    
    def compute(self, 
                retrieved_docs: List[str], 
                relevant_docs: Set[str],
                scores: Optional[List[float]] = None) -> EvaluationResult:
        """
        Compute NDCG@K for a single query.
        Uses binary relevance (1 for relevant, 0 for not relevant).
        """
        # Binary relevance: 1 if relevant, 0 if not
        relevance_scores = []
        top_k_docs = retrieved_docs[:self.k]
        
        for doc_id in top_k_docs:
            relevance_scores.append(1.0 if doc_id in relevant_docs else 0.0)
        
        # Compute DCG
        dcg = 0.0
        for i, relevance in enumerate(relevance_scores):
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / np.log2(i + 1)
        
        # Compute Ideal DCG (IDCG)
        ideal_relevance = sorted([1.0] * min(len(relevant_docs), self.k), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / np.log2(i + 1)
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return EvaluationResult(
            value=ndcg,
            metadata={
                "k": self.k,
                "dcg": dcg,
                "idcg": idcg,
                "relevant_in_top_k": sum(relevance_scores)
            }
        )


def create_synthetic_relevance_judgments(
    ctx_df: pd.DataFrame, 
    score_threshold: float = 0.3
) -> Set[str]:
    """
    Create synthetic relevance judgments based on search scores.
    
    Args:
        ctx_df: Context DataFrame with search results
        score_threshold: Minimum score to consider document relevant
        
    Returns:
        Set of document IDs considered relevant
    """
    logger.info(f"🔍 METRICS: Creating synthetic relevance judgments with threshold {score_threshold}")
    logger.info(f"🔍 METRICS: Processing {len(ctx_df)} documents")
    
    relevant_docs = set()
    
    for i, (_, row) in enumerate(ctx_df.iterrows()):
        metadata = row.get("metadata", {})
        logger.info(f"🔍 METRICS: Row {i} - metadata type: {type(metadata)}")
        
        # Handle both dict and JSON string metadata
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
                logger.info(f"🔍 METRICS: Row {i} - parsed JSON metadata successfully")
            except Exception as e:
                logger.warning(f"🔍 METRICS: Row {i} - failed to parse JSON metadata: {e}")
                metadata = {}
        
        # Get score from RRF or other hybrid search systems - try different score field names
        rrf_score = metadata.get("rrf_score")
        hybrid_score = metadata.get("hybrid_score")
        score_field = metadata.get("score")
        vector_score = metadata.get("vector_score")
        
        logger.info(f"🔍 METRICS: Row {i} - rrf_score: {rrf_score}, hybrid_score: {hybrid_score}, score: {score_field}, vector_score: {vector_score}")
        
        # Prioritize RRF score since that's what our system uses
        score = rrf_score or hybrid_score or score_field or vector_score or 0.0
        logger.info(f"🔍 METRICS: Row {i} - final score: {score}, threshold: {score_threshold}")
        
        if score >= score_threshold:
            doc_id = row.get("id", metadata.get("id", f"doc_{len(relevant_docs)}"))
            logger.info(f"🔍 METRICS: Row {i} - document {doc_id} is RELEVANT (score {score} >= {score_threshold})")
            relevant_docs.add(str(doc_id))
        else:
            doc_id = row.get("id", metadata.get("id", f"doc_{i}"))
            logger.info(f"🔍 METRICS: Row {i} - document {doc_id} is NOT RELEVANT (score {score} < {score_threshold})")
    
    logger.info(f"🔍 METRICS: Created synthetic relevance judgments: {len(relevant_docs)} relevant docs from {len(ctx_df)} total")
    if relevant_docs:
        logger.info(f"🔍 METRICS: Relevant doc IDs: {list(relevant_docs)}")
    else:
        logger.warning(f"🔍 METRICS: NO RELEVANT DOCUMENTS FOUND with threshold {score_threshold} - using fallback strategy!")
        
        # Fallback: Consider top 30% of documents as relevant if no documents meet threshold
        if len(ctx_df) > 0:
            fallback_count = max(1, len(ctx_df) // 3)  # At least 1, or top 1/3
            logger.info(f"🔍 METRICS: Fallback - marking top {fallback_count} documents as relevant")
            
            for i in range(min(fallback_count, len(ctx_df))):
                row = ctx_df.iloc[i]
                doc_id = row.get("id")
                if doc_id is None:
                    metadata = row.get("metadata", {})
                    if isinstance(metadata, str):
                        import json
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    doc_id = metadata.get("id", f"doc_fallback_{i}")
                
                relevant_docs.add(str(doc_id))
                logger.info(f"🔍 METRICS: Fallback - marked document {doc_id} as relevant")
    
    return relevant_docs


def evaluate_search_results(
    ctx_df: pd.DataFrame,
    query: str,
    relevance_judgments: Optional[Set[str]] = None,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, EvaluationResult]:
    """
    Evaluate search results using multiple IR metrics.
    
    Args:
        ctx_df: Context DataFrame with search results
        query: The search query (for logging)
        relevance_judgments: Set of relevant document IDs, if None will create synthetic
        k_values: List of K values for Precision@K, Recall@K, NDCG@K metrics
        
    Returns:
        Dictionary of metric names to EvaluationResult objects
    """
    logger.info(f"🔍 METRICS: evaluate_search_results called for query: {query[:50]}...")
    logger.info(f"🔍 METRICS: ctx_df is None: {ctx_df is None}")
    logger.info(f"🔍 METRICS: ctx_df length: {len(ctx_df) if ctx_df is not None else 'N/A'}")
    
    if ctx_df is None or len(ctx_df) == 0:
        logger.warning(f"🔍 METRICS: No search results to evaluate for query: {query}")
        return {}
    
    # Extract document IDs in ranking order
    retrieved_docs = []
    logger.info(f"🔍 METRICS: Extracting document IDs from {len(ctx_df)} rows...")
    
    for i, (_, row) in enumerate(ctx_df.iterrows()):
        doc_id = row.get("id")
        logger.info(f"🔍 METRICS: Row {i} - direct id field: {doc_id}")
        
        if doc_id is None:
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                    logger.info(f"🔍 METRICS: Row {i} - parsed metadata for id lookup")
                except:
                    logger.warning(f"🔍 METRICS: Row {i} - failed to parse metadata for id lookup")
                    metadata = {}
            doc_id = metadata.get("id", f"doc_{len(retrieved_docs)}")
            logger.info(f"🔍 METRICS: Row {i} - extracted id from metadata: {doc_id}")
        
        retrieved_docs.append(str(doc_id))
    
    logger.info(f"🔍 METRICS: Retrieved documents: {retrieved_docs}")
    
    # Create synthetic relevance judgments if none provided
    if relevance_judgments is None:
        logger.info(f"🔍 METRICS: No relevance judgments provided, creating synthetic ones...")
        relevance_judgments = create_synthetic_relevance_judgments(ctx_df)
    else:
        logger.info(f"🔍 METRICS: Using provided relevance judgments: {relevance_judgments}")
    
    if not relevance_judgments:
        logger.warning(f"🔍 METRICS: No relevant documents found for query: {query} - RETURNING EMPTY DICT!")
        return {}
    
    # Compute metrics
    results = {}
    
    # MRR
    mrr_metric = MeanReciprocalRank()
    results["mrr"] = mrr_metric.compute(retrieved_docs, relevance_judgments)
    
    # MAP
    map_metric = MeanAveragePrecision()
    results["map"] = map_metric.compute(retrieved_docs, relevance_judgments)
    
    # Precision@K, Recall@K, NDCG@K for different K values
    for k in k_values:
        if k <= len(retrieved_docs):
            # Precision@K
            precision_metric = PrecisionAtK(k)
            results[f"precision_at_{k}"] = precision_metric.compute(retrieved_docs, relevance_judgments)
            
            # Recall@K
            recall_metric = RecallAtK(k)
            results[f"recall_at_{k}"] = recall_metric.compute(retrieved_docs, relevance_judgments)
            
            # NDCG@K
            ndcg_metric = NDCGAtK(k)
            results[f"ndcg_at_{k}"] = ndcg_metric.compute(retrieved_docs, relevance_judgments)
    
    logger.info(f"Computed {len(results)} evaluation metrics for query: {query[:50]}...")
    return results