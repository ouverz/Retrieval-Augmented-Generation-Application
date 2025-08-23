from __future__ import annotations
import json
from datetime import datetime
from typing import List, Optional, Tuple, Union

import pandas as pd
from timescale_vector.client import uuid_from_time

from core.database.vector_store import VectorStore
from core.utils.extract_keywords import extract_keywords


class VectorSearchEngine:
    """Vector similarity via Timescale Vector (VectorStore)."""

    def __init__(
        self,
        chunked_texts: Optional[List[object]] = None,
        pdf_metadata: Optional[dict] = None,
        file_name: Optional[str] = None,
        file_path: Optional[str] = None,
        chunker: Optional[object] = None,
    ) -> None:
        self.vector_store = VectorStore()
        self.chunked_texts = list(chunked_texts) if chunked_texts else []
        self.pdf_metadata = pdf_metadata or {}
        self.file_name = file_name
        self.file_path = file_path
        self.chunker = chunker

    def create_embeddings(self) -> pd.DataFrame:
        cols = [
            "uuid_chunk",
            "metadata",
            "keywords",
            "chunk_text",
            "chunk_enriched",
            "embeddings",
        ]
        df = pd.DataFrame(columns=cols)
        keywords = {"keywords": []}

        for i, ch in enumerate(self.chunked_texts):
            text = getattr(ch, "text", "") or ""
            if not text:
                continue

            content = text
            if self.chunker and hasattr(self.chunker, "contextualize"):
                try:
                    content = self.chunker.contextualize(chunk=ch)
                except Exception:
                    content = text

            if i == 0:
                try:
                    keywords = extract_keywords(text) or {"keywords": []}
                except Exception:
                    keywords = {"keywords": []}

            emb = self.vector_store.get_embeddings(content)
            metadata = {
                "filename": self.file_name,
                "title": self.pdf_metadata.get("title"),
                "page_count": self.pdf_metadata.get("page_count"),
                "publishing_year": self.pdf_metadata.get("publishing_year"),
            }
            row = {
                "uuid_chunk": str(uuid_from_time(datetime.now())),
                "metadata": json.dumps(metadata, allow_nan=True),
                "keywords": json.dumps(keywords.get("keywords", [])),
                "chunk_text": text,
                "chunk_enriched": content,
                "embeddings": emb,
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        return df

    @staticmethod
    def apply_prepare_record(docs_df: pd.DataFrame) -> pd.DataFrame:
        cols = ["uuid_chunk", "keywords", "chunk_enriched", "embeddings"]
        missing = [c for c in cols if c not in docs_df.columns]
        if missing:
            raise ValueError(f"Missing columns for upsert: {missing}")
        return docs_df[cols].copy()

    def upsert_records(self, ready_records: pd.DataFrame) -> None:
        self.vector_store.drop_index()
        self.vector_store.create_tables()
        self.vector_store.create_index()
        self.vector_store.upsert(ready_records)

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Union[dict, List[dict], None] = None,
        predicates=None,
        time_range: Optional[Tuple] = None,
        return_dataframe: bool = True,
    ) -> Union[pd.DataFrame, list]:
        return self.vector_store.search(
            query_text=query,
            limit=top_k,
            metadata_filter=metadata_filter,
            predicates=predicates,
            time_range=time_range,
            return_dataframe=return_dataframe,
        )
