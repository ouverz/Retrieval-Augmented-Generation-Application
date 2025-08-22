# app/schemas/query.py
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from .common import SynthesizedResponse


class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, str]] = None
    top_k: int = Field(default=8, ge=1, le=100)


class QueryResponse(SynthesizedResponse):
    latency_ms: int
    results_table: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Detailed breakdown of search results with scores and rankings"
    )
