# app/schemas/query.py
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from .common import SynthesizedResponse
from .evaluation import EvaluationMetrics


class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, str]] = None
    top_k: int = Field(default=8, ge=1, le=100)
    session_id: Optional[str] = Field(default=None, description="Optional session ID for tracking")
    rrf_k: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=1000, 
        description="RRF k parameter for rank fusion (1-1000). Higher values make fusion more conservative"
    )
    enable_evaluation: bool = Field(
        default=False, 
        description="Enable evaluation metrics computation (may impact performance)"
    )


class QueryResponse(SynthesizedResponse):
    latency_ms: int
    results_table: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Detailed breakdown of search results with scores and rankings"
    )
    cache_hit: bool = Field(default=False, description="Whether response came from cache")
    cache_key: Optional[str] = Field(default=None, description="Cache key used (for debugging)")
    session_id: Optional[str] = Field(default=None, description="Session ID if provided")
    evaluation_metrics: Optional[EvaluationMetrics] = Field(
        default=None, 
        description="Evaluation metrics for search quality assessment"
    )
