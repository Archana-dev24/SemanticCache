import os
import numpy as np
import pandas as pd
import redis
from sentence_transformers import SentenceTransformer
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache

# --- Config ---
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.csv")
REDIS_HOST = "localhost"
REDIS_PORT = 6379
DISTANCE_THRESHOLD = 0.15
CACHE_TTL = 86400  # 24 hours
global cache
# --- Load data ---
df = pd.read_csv(CSV_PATH)[["question", "answer"]]

# --- Redis connection ---
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# --- Sentence encoder ---
encoder = SentenceTransformer("all-mpnet-base-v2")
faq_embeddings = encoder.encode(df["question"].tolist())




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
    for _, row in df.iterrows():
        cache.store(prompt=row["question"], response=row["answer"])
    cache.set_ttl(CACHE_TTL)
    return cache


if __name__ == "__main__":
    cache = build_redis_cache()

    print("=== Spot-check (3 queries) ===")
    test_queries = [
        "What activities can I do today?",
        "What are the best things to do in Abu Dhabi",
        "What are the top activities to explore?",
    ]
    for query in test_queries:
        result = check_cache(query)
        if result:
            print(f"  HIT : '{query}'")
            print(f"        {result[0]['response'][:70]}...")
            print(f"        distance={result[0].get('vector_distance', 'N/A')}")
        else:
            print(f"  MISS: '{query}'")
    print()

    print("=== RedisVL SemanticCache — extended queries ===")

    extended_queries = [
        "What activities can I do today?",
        "What can I do in Abu Dhabi today?",
        "What are some fun activities in Abu Dhabi?",
        "What are the best things to do in Abu Dhabi?",
        "What should I explore in Abu Dhabi today?",
        "What are the top activities to explore?",
        "What are the best experiences in Abu Dhabi?",
        "Which places should I visit today?",
        "Where should I go in Abu Dhabi today?",
        "Can you suggest places to visit in Abu Dhabi?",
        "What are the best places to visit in Abu Dhabi?",
        "What are Abu Dhabi's must-visit attractions?",
        "What are the top tourist attractions in Abu Dhabi?",
        "Which attractions should I not miss in Abu Dhabi?",
        "What can I see in Abu Dhabi today?",
        "What are the most popular places in Abu Dhabi?",
        "What are some interesting places to visit in Abu Dhabi?",
        "Can you recommend activities and places to visit in Abu Dhabi?",
        "How can I spend a day in Abu Dhabi?",
        "What are the top sightseeing spots in Abu Dhabi?",
    ]
    for q in extended_queries:
        result = cache.check(q)
        if result:
            print(f"  HIT : '{q}'")
            print(f"        {result[0]['response'][:70]}...")
        else:
            print(f"  MISS: '{q}'")
