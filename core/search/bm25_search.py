from __future__ import annotations
from typing import List
import logging
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
import re
from pathlib import Path

logger = logging.getLogger(__name__)

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization that doesn't require NLTK data downloads."""
    if not text:
        return []
    
    # Convert to lowercase and split on whitespace and punctuation
    text = text.lower()
    # Keep alphanumeric and handle contractions/hyphens
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
    return tokens

# Try to use NLTK if available, otherwise fall back to simple tokenizer
def get_nltk_tokenizer():
    """Get best available NLTK tokenizer."""
    try:
        from nltk.tokenize import word_tokenize
        # Test if word_tokenize works (requires punkt_tab in newer NLTK)
        word_tokenize("test")
        return word_tokenize, "word_tokenize"
    except (LookupError, OSError):
        # Fall back to TreebankWordTokenizer (no external data required)
        from nltk.tokenize import TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()
        return tokenizer.tokenize, "TreebankWordTokenizer"

try:
    import nltk
    
    # Set NLTK data path to project directory if it exists
    nltk_data_dir = Path(__file__).parent.parent.parent / "data" / "nltk_data"
    if nltk_data_dir.exists():
        nltk.data.path.insert(0, str(nltk_data_dir))
    
    # Get the best available NLTK tokenizer
    tokenizer_func, tokenizer_name = get_nltk_tokenizer()
    logger.info(f"Using NLTK {tokenizer_name} for BM25")
    
except (ImportError, Exception) as e:
    # Use simple tokenizer as fallback
    tokenizer_func = simple_tokenize
    logger.warning(f"Using simple tokenizer for BM25 (NLTK error: {type(e).__name__})")


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

                doc = Document(page_content=row["chunk_enriched"], metadata=meta)
                self.documents.append(doc)

        if not self.documents:
            raise ValueError("No documents to index for BM25.")
        self.retriever = BM25Retriever.from_documents(
            self.documents, preprocess_func=tokenizer_func
        )

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        if not self.retriever:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        self.retriever.k = top_k
        return self.retriever.invoke(query)
