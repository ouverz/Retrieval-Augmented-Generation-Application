from __future__ import annotations
import os
import pandas as pd
from typing import Optional

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from core.utils.extract_keywords import extract_metadata_with_pypdf
from core.search.vector_search import VectorSearchEngine
from core.search.bm25_search import BM25SearchEngine
from core.search.hybrid_search import HybridSearchEngine
from config.settings import HybridSearchConfig


class DocumentProcessor:
    """PDF → Docling → HybridChunker → vectors upsert + BM25 accumulation."""

    def __init__(
        self,
        directory: str,
        bm25_engine: BM25SearchEngine,
        vector_engine: VectorSearchEngine,
        chunker: Optional[HybridChunker] = None,
    ) -> None:
        self.directory = directory
        self.converter = DocumentConverter()
        self.chunker = chunker or HybridChunker(tokenizer="BAAI/bge-m3")
        self.bm25 = bm25_engine
        self.vector = vector_engine

    def process_directory(self) -> pd.DataFrame:
        all_docs = pd.DataFrame(
            columns=[
                "uuid_chunk",
                "metadata",
                "keywords",
                "chunk_text",
                "chunk_enriched",
                "embeddings",
            ]
        )
        pdfs = [f for f in os.listdir(self.directory) if f.lower().endswith(".pdf")]
        for fname in pdfs:
            df = self.process_single_pdf(os.path.join(self.directory, fname))
            if isinstance(df, pd.DataFrame) and not df.empty:
                all_docs = pd.concat([all_docs, df], ignore_index=True)
        return all_docs

    def process_single_pdf(self, file_path: str) -> pd.DataFrame:
        try:
            meta = extract_metadata_with_pypdf(file_path)
        except Exception:
            return pd.DataFrame()

        try:
            result = self.converter.convert(source=file_path)
            doc = result.document
        except Exception:
            return pd.DataFrame()

        try:
            chunks = list(self.chunker.chunk(doc))
            if not chunks:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

        try:
            vec_engine = VectorSearchEngine(
                chunks, meta, os.path.basename(file_path), file_path, self.chunker
            )
            df = vec_engine.create_embeddings()
            if isinstance(df, pd.DataFrame) and not df.empty:
                ready = VectorSearchEngine.apply_prepare_record(df)
                vec_engine.upsert_records(ready)
        except Exception:
            df = pd.DataFrame()

        try:
            self.bm25.add_chunks(file_path, chunks)
        except Exception:
            pass

        return df

    def run_application(self) -> pd.DataFrame:
        return self.process_directory()


class RAGApplication:
    """Orchestrates shared engines; finalize builds BM25 and Hybrid."""

    def __init__(self, config: HybridSearchConfig) -> None:
        self.config = config
        self.bm25_engine = BM25SearchEngine()
        self.vector_engine = VectorSearchEngine()
        self.hybrid_engine: Optional[HybridSearchEngine] = None

    def finalize(self) -> None:
        if self.bm25_engine.retriever is None:
            self.bm25_engine.build_index()
        self.hybrid_engine = HybridSearchEngine(
            self.bm25_engine, self.vector_engine, self.config
        )

    def search(self, query: str):
        if not self.hybrid_engine:
            raise RuntimeError("Call finalize() before search().")
        return self.hybrid_engine.search(query)
