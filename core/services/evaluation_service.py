"""
Evaluation service for RAG system performance assessment.
Provides real-time and cached evaluation of search results using IR metrics.
"""
import time
import logging
import hashlib
from typing import Dict, Any, Optional, Set, List
import pandas as pd
from datetime import datetime

from core.evaluation.metrics import evaluate_search_results, EvaluationResult
from backend.schemas.evaluation import (
    EvaluationMetrics, MetricResult, MetricQuality, 
    EvaluationSummary, EvaluationMetadata
)
from core.services.cache_service import CacheService

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for computing and caching evaluation metrics."""
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        """Initialize evaluation service with optional caching."""
        self.cache_service = cache_service
        self.evaluation_cache_ttl = 1800  # 30 minutes
        
        # Quality thresholds for metric interpretation
        self.quality_thresholds = {
            "mrr": {"excellent": 0.8, "good": 0.6, "fair": 0.4},
            "map": {"excellent": 0.7, "good": 0.5, "fair": 0.3},
            "precision": {"excellent": 0.8, "good": 0.6, "fair": 0.4},
            "recall": {"excellent": 0.8, "good": 0.6, "fair": 0.4},
            "ndcg": {"excellent": 0.8, "good": 0.6, "fair": 0.4}
        }
    
    def _get_cache_key(self, query: str, ctx_df_hash: str) -> str:
        """Generate cache key for evaluation results."""
        content = f"eval:{query}:{ctx_df_hash}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _hash_context_df(self, ctx_df: pd.DataFrame) -> str:
        """Generate hash for context DataFrame to detect changes."""
        if ctx_df is None or len(ctx_df) == 0:
            return "empty"
        
        # Create hash based on document IDs and scores
        content_parts = []
        for _, row in ctx_df.iterrows():
            doc_id = str(row.get("id", ""))
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            score = metadata.get("hybrid_score", 
                               metadata.get("score", 0.0))
            content_parts.append(f"{doc_id}:{score}")
        
        content = "|".join(content_parts)
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    
    def _interpret_quality(self, value: float, metric_type: str) -> MetricQuality:
        """Interpret metric value as quality level."""
        # Determine base metric type for threshold lookup
        if metric_type.startswith("precision_at_"):
            base_type = "precision"
        elif metric_type.startswith("recall_at_"):
            base_type = "recall"
        elif metric_type.startswith("ndcg_at_"):
            base_type = "ndcg"
        else:
            base_type = metric_type
        
        thresholds = self.quality_thresholds.get(base_type, self.quality_thresholds["map"])
        
        if value >= thresholds["excellent"]:
            return MetricQuality.EXCELLENT
        elif value >= thresholds["good"]:
            return MetricQuality.GOOD
        elif value >= thresholds["fair"]:
            return MetricQuality.FAIR
        else:
            return MetricQuality.POOR
    
    def _get_metric_description(self, metric_name: str) -> str:
        """Get human-readable description for metric."""
        descriptions = {
            "mrr": "Mean Reciprocal Rank - measures how well the system ranks the first relevant document",
            "map": "Mean Average Precision - overall precision across all relevant documents",
            "precision_at_1": "Precision@1 - accuracy of the top result",
            "precision_at_3": "Precision@3 - accuracy within top 3 results",
            "precision_at_5": "Precision@5 - accuracy within top 5 results",
            "precision_at_10": "Precision@10 - accuracy within top 10 results",
            "recall_at_1": "Recall@1 - fraction of relevant docs found in top 1 result",
            "recall_at_3": "Recall@3 - fraction of relevant docs found in top 3 results",
            "recall_at_5": "Recall@5 - fraction of relevant docs found in top 5 results",
            "recall_at_10": "Recall@10 - fraction of relevant docs found in top 10 results",
            "ndcg_at_1": "NDCG@1 - ranking quality considering position for top 1 result",
            "ndcg_at_3": "NDCG@3 - ranking quality considering position for top 3 results",
            "ndcg_at_5": "NDCG@5 - ranking quality considering position for top 5 results",
            "ndcg_at_10": "NDCG@10 - ranking quality considering position for top 10 results"
        }
        return descriptions.get(metric_name, f"Evaluation metric: {metric_name}")
    
    def _convert_to_metric_result(self, result: EvaluationResult, metric_name: str) -> MetricResult:
        """Convert EvaluationResult to MetricResult schema."""
        return MetricResult(
            value=result.value,
            confidence_interval=result.confidence_interval,
            interpretation=self._interpret_quality(result.value, metric_name),
            description=self._get_metric_description(metric_name)
        )
    
    def evaluate_query_results(
        self,
        query: str,
        ctx_df: pd.DataFrame,
        relevance_judgments: Optional[Set[str]] = None,
        use_cache: bool = True,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Optional[EvaluationMetrics]:
        """
        Evaluate search results and return structured metrics.
        
        Args:
            query: The search query
            ctx_df: Context DataFrame with search results
            relevance_judgments: Optional set of relevant document IDs
            use_cache: Whether to use cached results if available
            k_values: List of K values for Precision@K, Recall@K, NDCG@K
            
        Returns:
            EvaluationMetrics object or None if evaluation fails
        """
        try:
            logger.info(f"🔍 EVAL SERVICE: Starting evaluation for query: {query[:50]}...")
            logger.info(f"🔍 EVAL SERVICE: ctx_df is None: {ctx_df is None}")
            logger.info(f"🔍 EVAL SERVICE: ctx_df length: {len(ctx_df) if ctx_df is not None else 'N/A'}")
            
            if ctx_df is None or len(ctx_df) == 0:
                logger.warning(f"🔍 EVAL SERVICE: No results to evaluate for query: {query}")
                return None
            
            # Debug: Log context DataFrame structure
            logger.info(f"🔍 EVAL SERVICE: ctx_df columns: {list(ctx_df.columns)}")
            logger.info(f"🔍 EVAL SERVICE: Sample row data:")
            if len(ctx_df) > 0:
                sample_row = ctx_df.iloc[0]
                logger.info(f"🔍 EVAL SERVICE:   id: {sample_row.get('id', 'MISSING')}")
                logger.info(f"🔍 EVAL SERVICE:   metadata type: {type(sample_row.get('metadata', 'MISSING'))}")
                if 'metadata' in sample_row and sample_row['metadata']:
                    metadata = sample_row['metadata']
                    if isinstance(metadata, dict):
                        logger.info(f"🔍 EVAL SERVICE:   metadata keys: {list(metadata.keys())}")
                        logger.info(f"🔍 EVAL SERVICE:   hybrid_score: {metadata.get('hybrid_score', 'MISSING')}")
                        logger.info(f"🔍 EVAL SERVICE:   score: {metadata.get('score', 'MISSING')}")
                    else:
                        logger.info(f"🔍 EVAL SERVICE:   metadata content (first 200 chars): {str(metadata)[:200]}")
                else:
                    logger.info(f"🔍 EVAL SERVICE:   metadata: MISSING OR EMPTY")
            
            # Check cache first
            cached_result = None
            if use_cache and self.cache_service and self.cache_service.is_available():
                ctx_hash = self._hash_context_df(ctx_df)
                cache_key = self._get_cache_key(query, ctx_hash)
                
                try:
                    cached = self.cache_service.redis_client.get(f"eval_metrics:{cache_key}")
                    if cached:
                        import json
                        cached_data = json.loads(cached)
                        logger.info(f"Cache hit for evaluation: {query[:50]}...")
                        return EvaluationMetrics(**cached_data)
                except Exception as e:
                    logger.warning(f"Error retrieving cached evaluation: {e}")
            
            # Compute evaluation metrics
            logger.info(f"🔍 EVAL SERVICE: Computing evaluation metrics for query: {query[:50]}...")
            start_time = time.time()
            
            raw_results = evaluate_search_results(
                ctx_df=ctx_df,
                query=query,
                relevance_judgments=relevance_judgments,
                k_values=k_values
            )
            
            logger.info(f"🔍 EVAL SERVICE: Raw results returned: {raw_results is not None}")
            logger.info(f"🔍 EVAL SERVICE: Raw results type: {type(raw_results)}")
            logger.info(f"🔍 EVAL SERVICE: Raw results length: {len(raw_results) if raw_results else 'N/A'}")
            if raw_results:
                logger.info(f"🔍 EVAL SERVICE: Raw results keys: {list(raw_results.keys())}")
            
            if not raw_results:
                logger.warning(f"🔍 EVAL SERVICE: No evaluation results computed for query: {query}")
                return None
            
            # Convert to structured metrics
            evaluation_metrics = EvaluationMetrics()
            
            # MRR
            if "mrr" in raw_results:
                evaluation_metrics.mrr = self._convert_to_metric_result(
                    raw_results["mrr"], "mrr"
                )
            
            # MAP
            if "map" in raw_results:
                evaluation_metrics.map_score = self._convert_to_metric_result(
                    raw_results["map"], "map"
                )
            
            # Precision@K
            precision_at_k = {}
            for k in k_values:
                metric_name = f"precision_at_{k}"
                if metric_name in raw_results:
                    precision_at_k[k] = self._convert_to_metric_result(
                        raw_results[metric_name], metric_name
                    )
            if precision_at_k:
                evaluation_metrics.precision_at_k = precision_at_k
            
            # Recall@K
            recall_at_k = {}
            for k in k_values:
                metric_name = f"recall_at_{k}"
                if metric_name in raw_results:
                    recall_at_k[k] = self._convert_to_metric_result(
                        raw_results[metric_name], metric_name
                    )
            if recall_at_k:
                evaluation_metrics.recall_at_k = recall_at_k
            
            # NDCG@K
            ndcg_at_k = {}
            for k in k_values:
                metric_name = f"ndcg_at_{k}"
                if metric_name in raw_results:
                    ndcg_at_k[k] = self._convert_to_metric_result(
                        raw_results[metric_name], metric_name
                    )
            if ndcg_at_k:
                evaluation_metrics.ndcg_at_k = ndcg_at_k
            
            computation_time = time.time() - start_time
            logger.info(f"Evaluation completed in {computation_time:.3f}s")
            
            # Cache the result
            if use_cache and self.cache_service and self.cache_service.is_available():
                try:
                    import json
                    cache_data = evaluation_metrics.model_dump()
                    self.cache_service.redis_client.setex(
                        f"eval_metrics:{cache_key}",
                        self.evaluation_cache_ttl,
                        json.dumps(cache_data)
                    )
                    logger.debug("Cached evaluation metrics")
                except Exception as e:
                    logger.warning(f"Error caching evaluation metrics: {e}")
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return None
    
    def create_evaluation_summary(
        self,
        metrics: EvaluationMetrics,
        total_queries: int = 1,
        total_relevant_docs: Optional[int] = None
    ) -> EvaluationSummary:
        """
        Create a comprehensive evaluation summary for UI display.
        
        Args:
            metrics: Computed evaluation metrics
            total_queries: Total number of queries evaluated
            total_relevant_docs: Total number of relevant documents
            
        Returns:
            EvaluationSummary with insights and recommendations
        """
        # Determine overall quality based on key metrics
        key_scores = []
        if metrics.mrr:
            key_scores.append(metrics.mrr.value)
        if metrics.map_score:
            key_scores.append(metrics.map_score.value)
        if metrics.precision_at_k and 5 in metrics.precision_at_k:
            key_scores.append(metrics.precision_at_k[5].value)
        
        overall_score = sum(key_scores) / len(key_scores) if key_scores else 0.0
        overall_quality = self._interpret_quality(overall_score, "overall")
        
        # Generate insights
        key_insights = []
        strengths = []
        weaknesses = []
        recommendations = []
        
        # MRR analysis
        if metrics.mrr:
            if metrics.mrr.value >= 0.8:
                strengths.append("Excellent at ranking relevant documents first")
            elif metrics.mrr.value <= 0.3:
                weaknesses.append("Poor ranking of first relevant document")
                recommendations.append("Improve ranking algorithms or query understanding")
        
        # Precision analysis
        if metrics.precision_at_k:
            high_precision_count = sum(1 for m in metrics.precision_at_k.values() 
                                     if m.value >= 0.7)
            if high_precision_count >= 2:
                strengths.append("High precision across multiple K values")
            elif high_precision_count == 0:
                weaknesses.append("Low precision across all K values")
                recommendations.append("Review search relevance and ranking quality")
        
        # NDCG analysis
        if metrics.ndcg_at_k:
            avg_ndcg = sum(m.value for m in metrics.ndcg_at_k.values()) / len(metrics.ndcg_at_k)
            if avg_ndcg >= 0.7:
                strengths.append("Good ranking quality with position awareness")
            elif avg_ndcg <= 0.4:
                weaknesses.append("Poor ranking quality considering document positions")
                recommendations.append("Focus on improving document ordering and relevance scoring")
        
        # Generate key insights
        if overall_score >= 0.7:
            key_insights.append("System demonstrates strong overall performance")
        elif overall_score <= 0.4:
            key_insights.append("System performance needs significant improvement")
        else:
            key_insights.append("System shows moderate performance with room for improvement")
        
        if len(strengths) > len(weaknesses):
            key_insights.append("More strengths identified than weaknesses")
        elif len(weaknesses) > len(strengths):
            key_insights.append("Several areas for improvement identified")
        
        # Create metadata
        metadata = EvaluationMetadata(
            total_queries=total_queries,
            total_relevant_docs=total_relevant_docs or 0,
            avg_relevant_per_query=total_relevant_docs / total_queries if total_relevant_docs else 0,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            dataset_name="Synthetic relevance judgments"
        )
        
        return EvaluationSummary(
            overall_quality=overall_quality,
            metrics=metrics,
            metadata=metadata,
            key_insights=key_insights,
            recommendations=recommendations,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get evaluation cache statistics."""
        if not self.cache_service or not self.cache_service.is_available():
            return {"error": "Cache service not available"}
        
        try:
            eval_keys = self.cache_service.redis_client.keys("eval_metrics:*")
            return {
                "evaluation_cache_count": len(eval_keys),
                "cache_ttl_seconds": self.evaluation_cache_ttl
            }
        except Exception as e:
            return {"error": f"Failed to get evaluation cache stats: {e}"}
    
    def clear_evaluation_cache(self) -> bool:
        """Clear all evaluation cache entries."""
        if not self.cache_service or not self.cache_service.is_available():
            return False
        
        try:
            eval_keys = self.cache_service.redis_client.keys("eval_metrics:*")
            if eval_keys:
                self.cache_service.redis_client.delete(*eval_keys)
                logger.info(f"Cleared {len(eval_keys)} evaluation cache entries")
            return True
        except Exception as e:
            logger.error(f"Error clearing evaluation cache: {e}")
            return False


# Factory function following existing service patterns
def create_evaluation_service(cache_service: Optional[CacheService] = None) -> EvaluationService:
    """Create evaluation service with optional caching support."""
    return EvaluationService(cache_service=cache_service)