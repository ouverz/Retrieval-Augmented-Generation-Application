import re
import fitz  # PyMuPDF
from typing import List, Dict
from pypdf import PdfReader  # Alternative to PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_keywords(text: str) -> Dict[str, List[str]]:
    """Extract keyword-like metadata entries from text."""
    metadata = {"keywords": [], "terms": [], "categories": []}

    # Patterns to look for, case-insensitive
    patterns = {
        "keywords": r"(keywords|key words)\s*[:\-–]\s*(.+)",
        "terms": r"(terms|key terms)\s*[:\-–]\s*(.+)",
        "categories": r"(categories|topics)\s*[:\-–]\s*(.+)",
    }

    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for _, match in matches:
            # Split values on comma or semicolon
            values = [term.strip().lower() for term in re.split(r"[;,]", match)]
            metadata[key].extend(values)

    return metadata


def extract_metadata_with_pypdf(file_path):
    """Extract metadata using pypdf (lighter alternative to PyMuPDF)"""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            metadata = pdf_reader.metadata

            # Extract basic info
            doc_metadata = {
                "title": None,
                "authors": None,
                "subject": None,
                "publishing_year": None,
                "page_count": len(pdf_reader.pages),
            }

            if metadata:
                # Extract title
                doc_metadata["title"] = getattr(metadata, "title", None)

                # Extract author
                doc_metadata["authors"] = getattr(metadata, "author", None)

                # Extract subject
                doc_metadata["subject"] = getattr(metadata, "subject", None)

                # Extract year from creation or modification date
                for date_attr in ["creation_date", "modification_date"]:
                    date_val = getattr(metadata, date_attr, None)
                    if date_val:
                        try:
                            if hasattr(date_val, "year"):
                                doc_metadata["publishing_year"] = date_val.year
                                break
                            elif isinstance(date_val, str):
                                # Try to extract year from string
                                year_match = re.search(r"(\d{4})", str(date_val))
                                if year_match:
                                    doc_metadata["publishing_year"] = int(
                                        year_match.group(1)
                                    )
                                    break
                        except Exception:
                            continue

            # Use filename as fallback for title
            if not doc_metadata["title"]:
                doc_metadata["title"] = Path(file_path).stem

            return doc_metadata

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error extracting metadata with pypdf: {e}")
        return {
            "title": Path(file_path),
            "authors": None,
            "subject": None,
            "publishing_year": None,
            "page_count": None,
        }
