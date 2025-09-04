# app/schemas/evaluation.py
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MetricQuality(str, Enum):
    """Quality levels for metric interpretation."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"


class MetricResult(BaseModel):
    """Individual metric result with value and interpretation."""
    value: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Metric value between 0.0 and 1.0"
    )
    confidence_interval: Optional[List[float]] = Field(
        default=None,
        description="95% confidence interval as [lower_bound, upper_bound]"
    )
    interpretation: MetricQuality = Field(
        ..., 
        description="Human-readable quality interpretation"
    )
    description: str = Field(
        ..., 
        description="Detailed explanation of what this metric measures"
    )


class EvaluationMetrics(BaseModel):
    """Container for all evaluation metrics."""
    mrr: Optional[MetricResult] = Field(
        default=None,
        description="Mean Reciprocal Rank - measures ranking quality"
    )
    precision_at_k: Optional[Dict[int, MetricResult]] = Field(
        default=None,
        description="Precision@K for different K values (e.g., K=1,3,5,10)"
    )
    map_score: Optional[MetricResult] = Field(
        default=None,
        description="Mean Average Precision - overall retrieval quality"
    )
    recall_at_k: Optional[Dict[int, MetricResult]] = Field(
        default=None,
        description="Recall@K for different K values"
    )
    ndcg_at_k: Optional[Dict[int, MetricResult]] = Field(
        default=None,
        description="Normalized Discounted Cumulative Gain@K"
    )


class EvaluationMetadata(BaseModel):
    """Metadata about the evaluation process."""
    total_queries: int = Field(
        ..., 
        ge=0, 
        description="Total number of queries evaluated"
    )
    total_relevant_docs: int = Field(
        ..., 
        ge=0, 
        description="Total number of relevant documents across all queries"
    )
    avg_relevant_per_query: float = Field(
        ..., 
        ge=0.0, 
        description="Average number of relevant documents per query"
    )
    evaluation_timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp when evaluation was performed"
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Name of the evaluation dataset used"
    )


class EvaluationSummary(BaseModel):
    """Overall evaluation summary for UI display."""
    overall_quality: MetricQuality = Field(
        ..., 
        description="Overall system performance assessment"
    )
    metrics: EvaluationMetrics = Field(
        ..., 
        description="Detailed evaluation metrics"
    )
    metadata: EvaluationMetadata = Field(
        ..., 
        description="Evaluation process metadata"
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="Human-readable key insights from the evaluation"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for improvement"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Identified system strengths"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Identified areas for improvement"
    )