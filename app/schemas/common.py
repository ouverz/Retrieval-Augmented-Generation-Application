# app/schemas/common.py
from typing import Optional, List, Annotated
from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    source_uri: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None


class SynthesizedResponse(BaseModel):
    thought_process: Annotated[List[str], Field(min_length=1)]
    answer: str
    enough_context: bool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    citations: List[Citation] = Field(default_factory=list)
    precision: float = 0.0
    evidence_precision: str = "low"
