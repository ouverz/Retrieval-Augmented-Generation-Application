# NLTK Setup Guide

This guide explains how to properly set up NLTK for BM25 tokenization in both development and production environments.

## Problem
The BM25 search engine uses NLTK's `word_tokenize` for better text preprocessing, but NLTK requires downloading language data files like `punkt` and `punkt_tab`.

## Solution Overview
The system implements a **graceful fallback strategy**:
1. **Primary**: Use NLTK `word_tokenize` if available
2. **Fallback**: Use simple regex-based tokenizer if NLTK data missing
3. **No crashes**: Application works regardless of NLTK setup

## Local Development Setup

### Option 1: Quick Setup (Recommended)
```bash
# Install NLTK
uv add "nltk>=3.8.1"  # or pip install "nltk>=3.8.1"

# Download NLTK data (run once)
python scripts/setup_nltk.py
```

### Option 2: Manual Setup
```bash
# Install NLTK
uv add "nltk>=3.8.1"

# Download data manually
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Verification
```bash
# Test that BM25 import works
python -c "from core.search.bm25_search import BM25SearchEngine; print('Success')"
```

You should see either:
- `✅ Using NLTK word_tokenize for BM25` (ideal)
- `⚠️ Using simple tokenizer for BM25 (NLTK not available)` (fallback)

## Production Setup

### Docker Environment
For containerized deployments, add to your Dockerfile:

```dockerfile
# Install NLTK and download data
RUN pip install "nltk>=3.8.1"
RUN python infrastructure/docker/download_nltk_data.py
```

### Environment Variables
```bash
# Optional: Set custom NLTK data directory
export NLTK_DATA=/app/data/nltk_data
```

## How It Works

### Code Structure
The BM25 search engine (`core/search/bm25_search.py`) implements:

```python
# Try NLTK first
try:
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.data.find('tokenizers/punkt')
    tokenizer_func = word_tokenize
    print("✅ Using NLTK word_tokenize for BM25")
except (ImportError, LookupError):
    # Fallback to simple tokenizer
    tokenizer_func = simple_tokenize
    print("⚠️ Using simple tokenizer for BM25 (NLTK not available)")
```

### Simple Tokenizer Fallback
If NLTK is unavailable, the system uses a regex-based tokenizer:

```python
def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization that doesn't require NLTK."""
    text = text.lower()
    return re.findall(r"\b\w+(?:'\w+)?\b", text)
```

This handles:
- Lowercase conversion
- Word boundaries
- Contractions (e.g., "don't" → ["don't"])
- Alphanumeric tokens

## Performance Comparison

| Tokenizer | Quality | Setup | Dependencies |
|-----------|---------|-------|-------------|
| NLTK word_tokenize | High | Requires data download | NLTK + data files |
| Simple regex | Good | No setup | Built-in Python |

## Troubleshooting

### Issue: `LookupError: Resource punkt_tab not found`
**Solution**: Run the setup script:
```bash
python scripts/setup_nltk.py
```

### Issue: NLTK download hangs
**Solution**: The fallback tokenizer will activate automatically. Check logs for:
```
⚠️ Using simple tokenizer for BM25 (NLTK not available)
```

### Issue: Import errors in tests
**Solution**: Ensure tests run with the same environment:
```bash
# Before running tests
python scripts/setup_nltk.py
pytest
```

## Files Created/Modified

- **`core/search/bm25_search.py`**: Updated with fallback tokenizer
- **`core/utils/nltk_setup.py`**: NLTK data downloader utility
- **`scripts/setup_nltk.py`**: Development setup script
- **`infrastructure/docker/download_nltk_data.py`**: Production setup script
- **`requirements.txt`**: Added `nltk>=3.8.1`

This setup ensures the RAG system works reliably in all environments while providing optimal performance when NLTK is properly configured.