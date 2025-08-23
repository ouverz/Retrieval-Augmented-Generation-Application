from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Iterable, Annotated
import pandas as pd
from pydantic import BaseModel, Field
from core.services.llm_service import LLMFactory


class Citation(BaseModel):
    """Structured reference to a retrieved chunk."""

    doc_id: Optional[str] = Field(None, description="Document identifier")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    source_url: Optional[str] = Field(None, description="Filepath of the source")
    score: Optional[float] = Field(
        None, description="Retriever score (higher is better)"
    )


class SynthesizedResponse(BaseModel):
    """
    Final, validated response returned by the LLM synthesizer.
    pydantic-ai will enforce this shape and auto-retry if validation fails.
    """

    thought_process: Annotated[
        List[str],
        Field(
            min_length=1,
            description="High-level thoughts in bullet form, rather than raw Chain-of-Thought",  # , that the AI assistant had while synthesizing the answer
        ),
    ]
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )
    confidence: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="The level of confidence in the answer returned is accurate (relies solely on the information it has regarding the context) and addresses the user's question",
        ),
    ]
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations supporting key claims.",
    )
    precision: float = Field(
        description="describes how many documents/chunks in the result set are relevant."
    )
    evidence_precision: Literal["low", "medium", "high"] = Field(
        description="refers to the degree to which research findings are accurate and reliable, reflecting the true effect of an intervention or the true nature of a phenomenon"
    )


# -----------------------------
# System prompt
# -----------------------------
SYSTEM_PROMPT = """
    # Role and Purpose
    You are an cautious, citation-first RAG synthesizer for a research lab which specializes in baby sleep research.
    Your task is to synthesize a coherent and helpful answer based on the given user's question and 
    relevant context retrieved from a knowledge database.

    # Guidelines:
    1. Provide a clear and concise answer to the question.
    2. Use only the provided context to support your answer.
    3. If the context is missing, irrelevant or insufficient to answer, set enough_context=false and provide a brief next-step.
    4. Be transparent when there is insufficient information to fully answer the question, clearly state that.
    5. Provide concise thought_process bullets (no chain-of-thought).
    6. Include citations for specific claims.
    7. Do not make up or infer information not present in the provided context.
    8. Maintain a helpful and professional tone appropriate for customer service.
    
    # Output Format
    You MUST respond with a valid JSON object that matches this exact structure:
    {
        "thought_process": ["bullet point 1", "bullet point 2"],
        "answer": "your answer here",
        "enough_context": true or false,
        "confidence": 0.0 to 1.0,
        "citations": [{"doc_id": "id_from_context", "chunk_id": "id_from_context", "source_url": "source_from_metadata", "score": 0.8}],
        "precision": 0.0 to 1.0,
        "evidence_precision": "low" or "medium" or "high"
    }
    
    # Citation Instructions:
    - Extract doc_id and chunk_id from the context records
    - For source_url, use the "source" field from metadata, or construct from available metadata fields
    - Set score to reflect how well the citation supports the claim (0.0-1.0)
    - Only include citations for specific factual claims in your answer
    
    Review the question from the user:
    """

# Build a module-level default factory by asking LLMFactory to resolve the default provider from settings.
_DEFAULT_FACTORY = LLMFactory()


# -----------------------------
# Helpers
# -----------------------------
def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """
    Ensure all required columns exist so .to_json won't crash.
    Missing ones are added with empty values.
    """
    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = None
    return out


def _contextframe_to_json(
    context: pd.DataFrame,
    columns_to_keep: List[str],
) -> str:
    """
    Convert the context DataFrame to a compact JSON string (records).
    Uses indent=2 for readability; set to None if you prefer smaller payloads.
    """
    if context is None or len(context) == 0:
        return "[]"

    # Only keep columns that actually exist in the DataFrame
    available_columns = [col for col in columns_to_keep if col in context.columns]
    if not available_columns:
        # Fallback: use basic columns that should always be present
        available_columns = ["id", "content"] if "id" in context.columns and "content" in context.columns else list(context.columns)[:3]
    
    context = _ensure_columns(context, available_columns)
    return context[available_columns].to_json(orient="records", indent=2)


# -----------------------------
# Public API
# -----------------------------
async def synthesize_answer(
    query: str,
    context: pd.DataFrame,
    *,
    factory: LLMFactory = _DEFAULT_FACTORY,
    columns_to_keep: Sequence[str] = (
        "id",
        "content", 
        "metadata",
    ),
    max_attempts: Optional[int] = 2,
) -> SynthesizedResponse:
    """
    Build a pydantic-ai Agent via the LLMFactory and return a validated SynthesizedResponse.

    Parameters
    ----------
    factory : LLMFactory
        Your provider-aware factory that returns a pydantic-ai Agent.
    query : str
        The user's natural-language question.
    context : pd.DataFrame
        Retrieved passages/chunks with metadata. Expected columns are given by `columns_to_keep`.
    columns_to_keep : Sequence[str]
        Which context columns to serialize into the prompt (order matters).
    max_attempts : Optional[int]
        How many validation retries pydantic-ai should attempt if the model
        returns an invalid structure.

    Returns
    -------
    SynthesizedResponse
        A fully validated result adhering to the schema.
    """
    # Short-circuit if there is no context at all: avoid wasting tokens.
    if context is None or len(context) == 0:
        return SynthesizedResponse(
            thought_process=["No retrieved context available."],
            answer="I don't have enough supporting context to answer this question.",
            enough_context=False,
            confidence=0.0,
            citations=[],
            precision=0.0,
            evidence_precision="low"
        )

    context_json = _contextframe_to_json(context, columns_to_keep)
    
    user_message = f"""
    Question:
    {query}

    Retrieved Context (JSON records):
    {context_json}

    Instructions:
    - Answer strictly from the retrieved context.
    - If any part of the answer is not directly supported, set enough_context=false and list what is missing.
    - For citations, use the exact field names from the context records (id, metadata fields, etc.)
    - Return ONLY the fields of the SynthesizedResponse schema (no extra keys).
    """
    # Handle retries manually
    attempts = max_attempts or 2
    last_error = None
    
    for attempt in range(attempts):
        try:
            result: SynthesizedResponse = await factory.run_structured(
                result_type=SynthesizedResponse,
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
            )
            return result

        except Exception as e:
            last_error = e
            print(f"Synthesis attempt {attempt + 1} failed: {e}")
            if attempt < attempts - 1:
                print(f"Retrying... ({attempt + 2}/{attempts})")
    
    # All attempts failed
    print(f"Failed to synthesize message after {attempts} attempts: {last_error}")
    return SynthesizedResponse(
        thought_process=["Error during synthesis - all retry attempts failed"],
        answer="I encountered an error processing your request. Please try again.",
        enough_context=False,
        confidence=0.0,
        citations=[],
        precision=0.0,
        evidence_precision="low",
    )
