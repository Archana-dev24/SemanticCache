import os
import numpy as np
import pandas as pd
import redis
import time
from sentence_transformers import SentenceTransformer
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from groq import Groq
from evalmetrics import EvalMetrics
from dotenv import load_dotenv
load_dotenv()
# --- Config ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
DISTANCE_THRESHOLD = 0.15
CACHE_TTL = 86400  # 24 hours
global cache
# --- Load data ---

# --- Redis connection ---
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
client = Groq()
MODEL = "llama-3.3-70b-versatile"

# --- Sentence encoder ---
encoder = SentenceTransformer("all-mpnet-base-v2")

def check_cache(query: str) -> dict | None:
    """Return cached answer if a similar question exists within threshold, else None."""
    result = cache.check(query)
    if result!=[]:
        return result
        
    return None


def add_to_cache(question: str, answer: str) -> None:
    """Add a new Q&A pair to the in-memory FAQ store."""
    if question !=None and answer !=None :
        cache.store(
            prompt=question,
            response=answer
        )
        print(f"Added to cache: '{question}'")


def build_redis_cache() -> SemanticCache:
    """Populate a RedisVL SemanticCache from the FAQ dataframe."""
    langcache_embed = HFTextVectorizer(
        model="redis/langcache-embed-v1",
        cache=EmbeddingsCache(redis_client=r, ttl=3600),
    )
    cache = SemanticCache(
        name="faq-cache",
        vectorizer=langcache_embed,
        redis_client=r,
        distance_threshold=DISTANCE_THRESHOLD,
    )
    
    cache.set_ttl(CACHE_TTL)
    return cache


def call_llm(question: str) -> str:

    
    result = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
        {
            "role": "user",
            "content": """
        You are a helpful customer support assistant. Answer this customer question concisely and professionally:
        
        Question: {question}
        
        Provide a helpful response in 1-2 sentences. If you don't have specific information, give a general helpful response."""


        }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
    )
    return result.choices[0].message.content


def run_cache_eval(cache, test_questions, ground_truth, model="llama-3.3-70b-versatile"):

    assert len(test_questions) == len(ground_truth), \
        "❌ test_questions and ground_truth must be the same length!"

    metrics = EvalMetrics()
    metrics.total_questions = len(test_questions)
    metrics.ground_truth = ground_truth

    print(f"\n🚀 Starting Semantic Cache Evaluation")
    print(f"   Model    : {model}")
    print(f"   Questions: {len(test_questions)}")
    print("=" * 60)

    for i, (question, expected) in enumerate(zip(test_questions, ground_truth), 1):
        label = "HIT (expected)" if expected == 1 else "MISS (expected)"
        print(f"\n[{i}/{len(test_questions)}] [{label}]")
        print(f"  Q: '{question}'")

        # ── Cache Check ──────────────────────────────────────────
        t_start = time.perf_counter()
        cached_result = cache.check(question)
        t_cache = time.perf_counter() - t_start

        if cached_result:
            distance = cached_result[0].get("vector_distance", 0)
            result_label = "✅ TP" if expected == 1 else "❌ FP"
            metrics.record_hit(t_cache, expected)
            print(f"  → CACHE HIT  {result_label}  ({t_cache*1000:.2f} ms | distance: {distance:.3f})")
            print(f"    📋 {cached_result[0]['response'][:70]}...")

        else:
            result_label = "✅ TN" if expected == 0 else "❌ FN"
            metrics.record_miss(t_cache, expected)
            print(f"  → CACHE MISS {result_label}  ({t_cache*1000:.2f} ms)")
            print(f"    🤖 Calling LLM...", end=" ", flush=True)

            t_llm_start = time.perf_counter()
            llm_response = call_llm(question)
            t_llm = time.perf_counter() - t_llm_start

            metrics.record_llm(t_llm, question, llm_response)
            print(f"done ({t_llm*1000:.2f} ms)")
            print(f"    💬 {llm_response[:70]}...")
            cache.store(prompt=question, response=llm_response)
            print(f"    💾 Stored in cache")

    metrics.print_summary()
    return metrics


test_questions = [
    "what is process  to reset my password?",
    "I lost my passowrd. How can i get it back",
    "Why is the app stuck on a blank screen?",
    "My password is not working, how do I get a new one?",
    "Can you help me recover access to my account?",
    "The app is not loading, what should I do?",
    "Can I use the app on my mobile phone?",

    "The website is not opening on my browser, help!",
     "What payment methods do you accept?"  ,
    "How do I enable two-factor authentication?",
]

ground_truth=[1,1,0,1,1,0,0,0,0,0]  

if __name__ == "__main__":
    cache = build_redis_cache()

    print("=== Spot-check (queries) ===")
run_cache_eval(
    cache          = cache,
    test_questions = test_questions,
    ground_truth   = ground_truth,
    model          = "llama-3.3-70b-versatile"
)
