# Retrieval-Augmented Generation Application

Answer questions grounded in your own documents — not hallucinated from model weights. A production-ready RAG system built with hybrid search, intelligent caching, and a multi-LLM backend.

Built with modern Python AI tooling: **Pydantic AI**, **TimescaleDB + pgvector**, **Redis**, and support for both **OpenAI** and **Anthropic Claude**.

---

## Stack

| Layer | Technology |
|---|---|
| AI Framework | Pydantic AI |
| LLM Providers | OpenAI GPT-4o · Anthropic Claude |
| Vector Store | TimescaleDB + pgvector |
| Cache | Redis |
| Backend | FastAPI |
| Frontend | Streamlit |
| Search | Hybrid: BM25 + Vector similarity |
| Evaluation | RAGAS |
| Observability | Langfuse |

---

## 🚀 Quick Start

```bash
# 1. Clone and navigate to project
git clone https://github.com/ouverz/Retrieval-Augmented-Generation-Application.git
cd Retrieval-Augmented-Generation-Application

# 2. Install dependencies (choose one method)
# Option A: Using UV (recommended)
uv sync

# Option B: Using pip
pip install -e .
# OR: pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 4. Setup NLTK data (one-time setup)
python scripts/setup_nltk.py

# 5. Start the application
python start_app.py
```

**Access Points:**
- 🌐 **Frontend**: http://localhost:8501 (Streamlit UI)
- ⚡ **API**: http://localhost:8000 (FastAPI backend)
- 📚 **Docs**: http://localhost:8000/docs (API documentation)

---

## 🏗️ Architecture

```
Frontend (Streamlit) → Backend (FastAPI) → Core Logic → Infrastructure
                                       ↓
                         [Redis Cache] [TimescaleDB+pgvector] [OpenAI / Claude]
```

### Key Components

- **Frontend**: Clean Streamlit interface for document queries
- **Backend**: FastAPI service with comprehensive API endpoints
- **Core**: Business logic for document processing and hybrid search
- **Cache**: Redis-powered caching (99.8% query speedup)
- **Database**: TimescaleDB with pgvector for scalable vector storage

---

## ✨ Features

### Hybrid Search Engine
- **Vector Search**: Semantic similarity via OpenAI embeddings
- **BM25 Search**: Keyword-based relevance scoring
- **True Hybrid Fusion**: Intelligent combination of both approaches
- **Quality Scoring**: Citation-penalty algorithms for better results

### Performance Optimization
- **Redis Caching**:
  - 24h embedding cache (90% cost reduction)
  - 1h query result cache (99.8% latency improvement)
  - Session management and user history
- **Graceful Degradation**: Full functionality even when cache unavailable

### Production Features
- **Structured Responses**: Type-safe API responses with citations (via Pydantic AI)
- **Health Monitoring**: Comprehensive system health checks
- **Error Handling**: Robust error recovery and logging
- **Docker Integration**: One-command infrastructure deployment

### 🧪 RAGAS Quality Evaluation

Automated scoring of every RAG response across four metrics (0.0–1.0 scale):

| Metric | What it measures | Ground truth needed? |
|--------|-----------------|---------------------|
| Faithfulness | Answer stays within retrieved context | No |
| Answer Relevancy | Answer addresses the question | No |
| Context Precision | Retrieved chunks are relevant to the answer | Yes |
| Context Recall | All necessary chunks were retrieved | Yes |

Scores are exposed via REST API and a benchmark CLI. Powered by `gpt-4o-mini` as judge — results are LRU-cached to avoid redundant API calls.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "answer": "...", "contexts": ["..."]}'
```

See **[docs/RAGAS.md](docs/RAGAS.md)** for the full API reference, Python SDK usage, benchmark CLI, and Langfuse integration.

---

## 📊 Performance

| Operation | Cold Cache | Warm Cache | Improvement |
|---|---|---|---|
| Document Query | 2–5 seconds | 10–50ms | 99.8% |
| Embedding Generation | 2–3 seconds | Instant | 99.9% |
| Search Processing | 500ms–2s | 16ms | 98%+ |

---

## 📁 Project Structure

```
rag-application/
├── config/           # Configuration management
├── frontend/         # Streamlit UI
├── backend/          # FastAPI service
├── core/             # Business logic
│   ├── processors/   # Document processing
│   ├── search/       # Search engines
│   ├── services/     # Core services
│   └── database/     # Data layer
├── infrastructure/   # Docker & deployment
├── scripts/          # Setup and utility scripts
├── data/             # Document storage
└── docs/             # Documentation
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
TIMESCALE_SERVICE_URL=postgres://postgres:your_secure_password@127.0.0.1:5432
REDIS_URL=redis://localhost:6379/0

# Optional
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=
```

### System Settings
- **Embedding Model**: text-embedding-3-small (1536 dimensions)
- **LLM**: GPT-4o for response generation
- **Search Weights**: 30% BM25 + 70% Vector (configurable)
- **Cache Policies**: Intelligent TTL management

---

## 🛠️ Development

### API Usage

```python
import requests

# Initialize system
requests.post("http://localhost:8000/init", json={"force": False})

# Query documents
response = requests.post("http://localhost:8000/query",
                         json={"query": "your question", "top_k": 5})
```

### Cache Management

```bash
# View cache statistics
curl http://localhost:8000/cache/stats

# Clear specific cache
curl -X POST http://localhost:8000/cache/clear?cache_type=embeddings

# Health check
curl http://localhost:8000/cache/health
```

---

## 🎯 Common Operations

### Document Management
1. **Add PDFs**: Place files in `data/documents/`
2. **Re-initialize**: POST to `/init` with `{"force": true}`
3. **Monitor**: Check `/health` endpoint for system status

### Search Configuration
- Modify `config/settings.py` for hybrid search weights
- Adjust cache TTL settings in Redis configuration
- Update embedding models and LLM preferences

---

## 🧪 Testing

```bash
# Full test suite
pytest

# Specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/validation/    # System validation
```

---

## 🔧 Troubleshooting

### Common Issues

**pgvector not found**
```bash
docker exec timescaledb psql -U postgres -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Redis connection failed**
```bash
docker-compose ps  # Check Redis container status
```

**Import errors after restructure**
```bash
pip install -r requirements.txt  # Reinstall dependencies
```

**Slow performance**
- Check Redis cache hit rates via `/cache/stats`
- Verify OpenAI API key limits
- Monitor TimescaleDB index status

---

## 📖 Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: Comprehensive system design
- **[API Documentation](docs/API.md)**: Complete API reference
- **[RAGAS Evaluation](docs/RAGAS.md)**: Evaluation API, benchmark CLI, Langfuse integration
- **[NLTK Setup](docs/NLTK_SETUP.md)**: NLTK configuration guide

---

## 🗺️ Roadmap

- [x] Hybrid search (BM25 + vector + RRF fusion)
- [x] Redis caching with graceful degradation
- [x] RAGAS evaluation API and benchmark CLI
- [x] Langfuse observability integration
- [x] API key authentication and security headers
- [ ] Multi-modal document support (images, tables)
- [ ] Advanced search filters and faceting
- [ ] Distributed caching with Redis Cluster
- [ ] Metrics and monitoring dashboard
- [ ] Kubernetes deployment manifests

---

## 📄 License

MIT License — see LICENSE file for details.

---

Built by [Ofer Kulka](https://www.oferkulka.com) — Senior Data & AI Engineer, Frankfurt am Main.  
[LinkedIn](https://linkedin.com/in/oferkulka) · [GitHub](https://github.com/ouverz)
