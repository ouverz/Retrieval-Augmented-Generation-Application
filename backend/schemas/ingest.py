# app/schemas/ingest.py
from pydantic import BaseModel


class InitRequest(BaseModel):
    force: bool = False


class InitResponse(BaseModel):
    initialized: bool
    reason: str


class InitStatusResponse(BaseModel):
    status: str  # "idle", "running", "completed", "error"
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    documents_processed: int = 0
    total_documents: int = 0


class IngestResponse(BaseModel):
    accepted: bool
    detail: str
