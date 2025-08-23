from __future__ import annotations
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document


class BM25SearchEngine:
    """BM25 keyword search over accumulated LangChain Documents."""

    def __init__(self) -> None:
        self.retriever: BM25Retriever | None = None
        self.documents: List[Document] = []

    def add_chunks(self, pdf_path: str, chunks: List[object]) -> None:
        if not isinstance(chunks, list):
            chunks = list(chunks)
        start = len(self.documents)
        for i, ch in enumerate(chunks, start=start):
            text = getattr(ch, "text", None)
            if not text:
                continue
            meta = {
                "source": pdf_path,
                "chunk_id": i,
                "section": getattr(ch, "section", None),
                "page": getattr(ch, "page", None),
            }
            self.documents.append(Document(page_content=text, metadata=meta))

    def build_index(self, docs_df=None) -> None:
        """Build BM25 index from documents DataFrame or existing documents list"""
        
        if docs_df is not None:
            # Use DataFrame with proper UUIDs (preferred method)
            import pandas as pd
            self.documents = []
            for _, row in docs_df.iterrows():
                meta = {
                    "id": row["uuid_chunk"],  # Use proper UUID
                    "source": row.get("file_name", ""),
                    "title": row.get("title", ""),
                    "author": row.get("author", ""),
                }
                # Add other metadata if present
                if pd.notna(row.get("keywords")):
                    meta["keywords"] = row["keywords"]
                
                doc = Document(
                    page_content=row["chunk_text"],
                    metadata=meta
                )
                self.documents.append(doc)
        
        if not self.documents:
            raise ValueError("No documents to index for BM25.")
        self.retriever = BM25Retriever.from_documents(self.documents)

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        if not self.retriever:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        self.retriever.k = top_k
        return self.retriever.invoke(query)
