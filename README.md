# SemanticCache

A semantic caching layer for LLM applications built on **Redis** and **sentence-transformers**. Instead of calling an LLM for every user question, semantically similar questions are served directly from cache — reducing latency and API costs.

---

## How It Works

```
User Query
    │
    ▼
Embed query  ──►  Search Redis SemanticCache
                        │
              ┌─────────┴─────────┐
           Cache HIT           Cache MISS
              │                    │
        Return cached          Call LLM
         response           Store response
                                   │
                            Return response
```

A cosine distance threshold (default `0.15`) determines whether an incoming question is "close enough" to a cached one. Questions within the threshold are answered instantly from cache; others fall through to the LLM and their response is stored for future reuse.

---

## Project Structure

| File | Purpose |
|------|---------|
| `semantic_cache.py` | Core cache layer — loads a FAQ CSV into Redis SemanticCache and exposes `check_cache` / `add_to_cache` helpers |
| `semantic_cache_llm.py` | LLM-integrated layer — on cache miss, calls Groq (Llama 3.3 70B) and auto-stores the response |
| `evalmetrics.py` | Evaluation harness — runs labelled test queries and reports precision, recall, F1, accuracy, and latency |
| `test.csv` | Sample FAQ dataset (Abu Dhabi tourism Q&A) |
| `requirements.txt` | Python dependencies |

---

## Prerequisites

- Python 3.10+
- Redis running locally on `localhost:6379`
  ```bash
  # Docker (quickest)
  docker run -d -p 6379:6379 redis/redis-stack
  ```
- A Groq API key (only needed for `semantic_cache_llm.py`)

---

## Installation

```bash
git clone https://github.com/Archana-dev24/SemanticCache.git
cd SemanticCache
pip install -r requirements.txt
```

Create a `.env` file for your API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Usage

### 1. FAQ Semantic Cache (`semantic_cache.py`)

Loads Q&A pairs from `test.csv` into Redis and answers incoming questions semantically.

```python
from semantic_cache import build_redis_cache, check_cache, add_to_cache

cache = build_redis_cache()          # load FAQ into Redis

result = check_cache("What can I do in Abu Dhabi?")
if result:
    print(result[0]["response"])     # cache hit
else:
    print("Cache miss — call your LLM here")

add_to_cache("New question?", "New answer.")   # extend the cache at runtime
```

Run the built-in spot-check and extended query test:

```bash
python semantic_cache.py
```

Sample output:

```
=== Spot-check (3 queries) ===
  HIT : 'What activities can I do today?'
        Abu Dhabi is bursting with adventures — beaches, wildlife, culture...
        distance=1.19e-07

=== RedisVL SemanticCache — extended queries ===
  HIT : 'What can I do in Abu Dhabi today?'
  HIT : 'What are the best things to do in Abu Dhabi?'
  MISS: 'What are the most popular places in Abu Dhabi?'
  ...
```

---

### 2. LLM-Integrated Cache with Evaluation (`semantic_cache_llm.py`)

On a cache miss, calls Groq's Llama 3.3 70B, stores the response, and evaluates cache quality against ground-truth labels.

```bash
python semantic_cache_llm.py
```

Sample output:

```
[1/10] [HIT (expected)]
  Q: 'what is process to reset my password?'
  → CACHE HIT  ✅ TP  (4.21 ms | distance: 0.082)

[2/10] [HIT (expected)]
  Q: 'I lost my password. How can I get it back'
  → CACHE HIT  ✅ TP  (3.87 ms | distance: 0.103)

[3/10] [MISS (expected)]
  Q: 'Why is the app stuck on a blank screen?'
  → CACHE MISS ✅ TN  (2.10 ms)
    🤖 Calling LLM... done (812.34 ms)

============================================================
       📊  SEMANTIC CACHE EVALUATION REPORT
============================================================
  📦 Cache Performance
  Total Questions               : 10
  Cache Hits                    : 4  (40.0%)
  Cache Misses                  : 6  (60.0%)

  🎯 Classification Metrics
  Precision                     : 1.0000  (100.0%)
  Recall                        : 0.8000  (80.0%)
  F1 Score                      : 0.8889  (88.9%)
  Accuracy                      : 0.9000  (90.0%)

  ⏱️  Latency
  Avg Cache Hit Time            : 4.05 ms
  Avg LLM Call Time             : 834.12 ms
  Est. Time Saved               : 3336.48 ms
============================================================
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DISTANCE_THRESHOLD` | `0.15` | Cosine distance cutoff — lower = stricter matching |
| `CACHE_TTL` | `86400` | Cache entry TTL in seconds (24 hours) |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `MODEL` | `llama-3.3-70b-versatile` | Groq model for LLM fallback |

Tune `DISTANCE_THRESHOLD` based on your evaluation results:
- **Too many False Positives** (wrong cached answer returned) → lower the threshold
- **Too many False Negatives** (missed obvious cache hits) → raise the threshold

---

## Embeddings

| Model | Used for |
|-------|----------|
| `all-mpnet-base-v2` | Cosine distance spot-checks (sentence-transformers) |
| `redis/langcache-embed-v1` | RedisVL SemanticCache vector index |

---

## Tech Stack

- [Redis Stack](https://redis.io/docs/stack/) — vector store and TTL-based cache
- [RedisVL](https://github.com/redis/redis-vl-python) — SemanticCache and vector index abstractions
- [sentence-transformers](https://www.sbert.net/) — local embedding models
- [Groq](https://groq.com/) — fast LLM inference (Llama 3.3 70B)
