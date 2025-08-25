# app/schemas/query.py
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from .common import SynthesizedResponse


class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, str]] = None
    top_k: int = Field(default=8, ge=1, le=100)
    session_id: Optional[str] = Field(default=None, description="Optional session ID for tracking")
    vector_weight: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Weight for vector search component (0.0-1.0). BM25 weight = 1.0 - vector_weight"
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
