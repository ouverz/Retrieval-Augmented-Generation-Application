# RAGAS Evaluation Guide

Automated quality scoring for RAG responses using [RAGAS](https://docs.ragas.io) with Langfuse observability.

---

## Metrics

All scores are on a **0.0–1.0 scale**. Higher is better.

| Metric | What it measures | Ground truth needed? |
|--------|-----------------|---------------------|
| **Faithfulness** | Answer stays within the retrieved context — no hallucinations | No |
| **Answer Relevancy** | Answer directly addresses the question | No |
| **Context Precision** | Retrieved chunks are relevant to the expected answer | Yes (`reference`) |
| **Context Recall** | All necessary information was retrieved | Yes (`reference`) |

Omit `reference` (or pass `""`) to score only faithfulness and answer relevancy.

---

## REST API

All endpoints require the `X-API-Key` header when API keys are configured.

### Check availability

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/evaluate/status
```

```json
{"available": true, "module": "ragas", "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
```

### Evaluate a single response

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "question": "What are the benefits of bedtime routines?",
    "answer": "Bedtime routines improve sleep duration and reduce onset latency.",
    "contexts": [
      "A nightly bedtime routine was associated with improved sleep outcomes.",
      "Children with routines showed better emotional regulation."
    ],
    "reference": "Consistent bedtime routines improve sleep outcomes in young children.",
    "use_cache": true
  }'
```

```json
{
  "scores": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.87,
    "context_precision": 0.85,
    "context_recall": 0.78,
    "overall": 0.855
  },
  "question": "What are the benefits of bedtime routines?",
  "answer_preview": "Bedtime routines improve sleep duration...",
  "num_contexts": 2,
  "evaluation_time_ms": 3240,
  "metadata": {"cache_hit": false}
}
```

### Evaluate a batch

Samples run concurrently via `asyncio.gather`.

```bash
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "samples": [
      {"question": "q1", "answer": "a1", "contexts": ["c1"], "reference": "r1"},
      {"question": "q2", "answer": "a2", "contexts": ["c2"], "reference": "r2"}
    ]
  }'
```

Response includes per-sample results and an aggregate `summary` block with averages across all metrics.

---

## Python API

```python
import asyncio
from core.evaluation.evaluator import RAGASEvaluator

evaluator = RAGASEvaluator()

# Single evaluation
result = asyncio.run(evaluator.evaluate(
    question="What causes poor sleep in toddlers?",
    answer="Irregular bedtime routines are a leading cause.",
    contexts=["Studies show irregular routines disrupt sleep patterns."],
    reference="Inconsistent bedtime routines are associated with poor sleep.",
))
print(result.scores.to_dict())
# {'faithfulness': 0.91, 'answer_relevancy': 0.88, 'context_precision': 0.84, 'context_recall': 0.79}

# Batch evaluation
results = asyncio.run(evaluator.evaluate_batch([
    {"question": "q1", "answer": "a1", "contexts": ["c1"], "reference": "r1"},
    {"question": "q2", "answer": "a2", "contexts": ["c2"], "reference": "r2"},
]))
summary = evaluator.get_summary(results)
print(summary["avg_overall"])
```

---

## Benchmark CLI

Queries the live RAG system and evaluates every response with RAGAS.

**Requirements:** application must be running (`python start_app.py`) and initialised (`POST /init`).

```bash
# Generate a test dataset from your documents (uses gpt-4o-mini)
python scripts/benchmark_ragas.py \
  --generate \
  --docs data/documents/ \
  --size 20

# Run the benchmark against an existing dataset
python scripts/benchmark_ragas.py \
  --dataset data/test_dataset.json \
  --top-k 5 \
  --output results/benchmark.json

# Generate and run in one command
python scripts/benchmark_ragas.py \
  --generate --docs data/documents/ --size 20 \
  --output results/benchmark.json
```

Example output:

```
============================================================
RAGAS BENCHMARK RESULTS
============================================================
Timestamp:     2026-06-01 00:00:00
RAGAS version: 0.4.3
Questions:     20 | Success: 19 | Failed: 1
------------------------------------------------------------
Metric                     Avg    Min    Max
------------------------------------------------------------
Faithfulness             0.871  0.620  1.000
Answer Relevancy         0.834  0.710  0.950
Context Precision        0.812  0.500  1.000
Context Recall           0.756  0.430  0.920
------------------------------------------------------------
Overall                  0.818
Avg eval time            3140ms
============================================================
```

---

## Caching

Evaluation results are cached in-process with a bounded LRU cache (512 entries), keyed on question + answer. This avoids redundant LLM-as-judge calls for repeated evaluations.

Pass `"use_cache": false` in the request body to bypass.

---

## Langfuse Observability

When `LANGFUSE_ENABLED=true`, every evaluation call is traced and RAGAS scores are pushed into the Langfuse timeline. The app runs fully without it.

```bash
# .env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | Used by RAGAS as the LLM judge |
| `LANGFUSE_ENABLED` | `false` | Enable/disable Langfuse tracing |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse project secret key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse instance URL |

The judge model is `gpt-4o-mini` (faster and cheaper than `gpt-4o`, configured in `config/settings.py`).

---

## Troubleshooting

**Evaluation returns zeros**
- `GET /evaluate/status` must return `"available": true`
- Run `uv sync` to confirm `ragas` and `langchain-openai` are installed
- Verify `OPENAI_API_KEY` is set

**`context_precision` / `context_recall` are 0.0**
- These metrics require a non-empty `reference` (ground truth answer)

**Slow evaluation**
- Each metric makes one OpenAI API call (~2–5 s total per evaluation)
- Enable caching (`"use_cache": true`) to skip repeated pairs
- Use the batch endpoint for multiple evaluations — they run concurrently
