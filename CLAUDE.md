# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) application using TimescaleDB with pgvector for document storage and similarity search. The system processes PDF documents, creates embeddings, and provides both BM25 keyword search and vector similarity search through a hybrid search engine.

## Architecture

The project follows a **client-server architecture**:

1. **FastAPI Backend** (`app/`): Core service handling all RAG operations
   - `app/main.py`: FastAPI application with health check and routers
   - `app/routers/`: API endpoints for initialization, ingestion, and querying
   - `app/schemas/`: Pydantic models for request/response validation
   - `app/services/`: Application container and state management

2. **Streamlit Frontend** (`app.py`): Pure frontend client that communicates with FastAPI backend
   - Makes HTTP requests to FastAPI endpoints
   - Provides user interface for system initialization and querying
   - No direct document processing or database access

## Core Components

- **DocumentProcessor** (`src/Processing_Documents.py`): Handles PDF processing using Docling, chunking with HybridChunker, and document ingestion
- **Search Engines**:
  - `VectorSearchEngine.py`: Semantic search using TimescaleDB pgvector
  - `BM25SearchEngine.py`: Keyword-based search using rank-bm25
  - `HybridSearchEngine.py`: Combines both approaches with configurable weights
- **Configuration** (`src/config/settings.py`): Centralized settings using Pydantic for OpenAI, database, and search parameters
- **Database** (`src/database/vector_store.py`): TimescaleDB integration for vector storage

## Environment Setup

Required environment variables in `.env`:
- `OPENAI_API_KEY`: For embeddings and LLM responses
- `TIMESCALE_SERVICE_URL`: TimescaleDB connection string

## Common Commands

### Database Setup
```bash
cd src/docker && docker-compose up -d
```

### Start Both Servers (Recommended)
```bash
python start_servers.py
```

### Start FastAPI Backend (Manual)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Streamlit Frontend (Manual)
```bash
streamlit run app.py
```

### Initialize System (via API or UI)
```bash
curl -X POST "http://localhost:8000/init" -H "Content-Type: application/json" -d '{"force": false}'
```

### Install Dependencies
```bash
pip install -r requirements.txt
# or using uv
uv sync
```

### Run Tests
```bash
pytest
```

## Document Processing Flow

1. PDFs in `/data` directory are processed using Docling DocumentConverter
2. Documents are chunked using HybridChunker with BAAI/bge-m3 tokenizer
3. Chunks are embedded using OpenAI text-embedding-3-small
4. Embeddings stored in TimescaleDB with metadata
5. BM25 index built for keyword search
6. Hybrid search combines both approaches using configurable weights

## Key Configuration

- Default embedding model: `text-embedding-3-small` (1536 dimensions)
- Default LLM: `gpt-4o`
- Hybrid search weights: 50% BM25, 50% vector (configurable in `HybridSearchConfig`)
- Vector table: `embeddings` with 7-day time partitioning

## Data Directory Structure

- `/data`: PDF documents for processing
- `processed_rag_documents.csv`: Processed document metadata
- `old_src/`: Legacy implementation for reference