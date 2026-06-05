import time
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# 1. Metrics Tracker (replaces perf_eval)
# ─────────────────────────────────────────────

@dataclass
class EvalMetrics:
    cache_hits:        int   = 0
    cache_misses:      int   = 0
    total_questions:   int   = 0

    cache_hit_times:   list  = field(default_factory=list)
    cache_miss_times:  list  = field(default_factory=list)
    llm_call_times:    list  = field(default_factory=list)

    llm_calls:         list  = field(default_factory=list)  # (question, response)
    # Confusion matrix components
    TP: int = 0   # Expected HIT  → Got HIT   ✅
    TN: int = 0   # Expected MISS → Got MISS  ✅
    FP: int = 0   # Expected MISS → Got HIT   ❌ (false cache hit)
    FN: int = 0   # Expected HIT  → Got MISS  ❌ (missed cache hit)
    predictions:  list = field(default_factory=list)   # actual cache result (1/0)
    ground_truth: list = field(default_factory=list)   # expected result     (1/0)
    def record_hit(self, elapsed: float, expected: int):
        self.cache_hits += 1
        self.cache_hit_times.append(elapsed)
        self.predictions.append(1)
        if expected == 1:
            self.TP += 1   # Correctly identified as HIT
        else:
            self.FP += 1   # Incorrectly identified as HIT (false positive)

    def record_miss(self, elapsed: float, expected: int):
        self.cache_misses += 1
        self.cache_miss_times.append(elapsed)
        self.predictions.append(0)
        if expected == 0:
            self.TN += 1   # Correctly identified as MISS
        else:
            self.FN += 1   # Incorrectly identified as MISS (false negative)

    def record_llm(self, elapsed: float, question: str, response: str):
        self.llm_call_times.append(elapsed)
        self.llm_calls.append({"question": question, "response": response})

    # ── Core Metrics ──────────────────────────────────────────

    @property
    def precision(self) -> float:
        """Of all cache HITs returned, how many were correct?"""
        denom = self.TP + self.FP
        return self.TP / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Of all expected HITs, how many did we actually catch?"""
        denom = self.TP + self.FN
        return self.TP / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Harmonic mean of Precision and Recall."""
        denom = self.precision + self.recall
        return 2 * (self.precision * self.recall) / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / total if total > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total) * 100 if total > 0 else 0.0


    @property
    def avg_hit_time(self) -> float:
        return sum(self.cache_hit_times) / len(self.cache_hit_times) if self.cache_hit_times else 0.0

    @property
    def avg_miss_time(self) -> float:
        return sum(self.cache_miss_times) / len(self.cache_miss_times) if self.cache_miss_times else 0.0

    @property
    def avg_llm_time(self) -> float:
        return sum(self.llm_call_times) / len(self.llm_call_times) if self.llm_call_times else 0.0

    @property
    def total_time_saved(self) -> float:
        # Estimate: each cache hit saved an avg LLM call time
        return self.cache_hits * self.avg_llm_time if self.llm_call_times else 0.0

    def print_summary(self):
        total = self.TP + self.TN + self.FP + self.FN
        print("\n" + "=" * 60)
        print("       📊  SEMANTIC CACHE EVALUATION REPORT")
        print("=" * 60)

        # Cache counts
        print(f"\n  📦 Cache Performance")
        print(f"  {'Total Questions':<30}: {total}")
        print(f"  {'Cache Hits':<30}: {self.cache_hits}  ({self.hit_rate:.1f}%)")
        print(f"  {'Cache Misses':<30}: {self.cache_misses}  ({100 - self.hit_rate:.1f}%)")

        # Confusion matrix
        print(f"\n  🔢 Confusion Matrix")
        print(f"  ┌─────────────────────────────────┐")
        print(f"  │           Predicted              │")
        print(f"  │        HIT        MISS           │")
        print(f"  │ Act HIT  TP={self.TP:<4}    FN={self.FN:<4}       │")
        print(f"  │ Act MISS FP={self.FP:<4}    TN={self.TN:<4}       │")
        print(f"  └─────────────────────────────────┘")

        # Classification metrics
        print(f"\n  🎯 Classification Metrics")
        print(f"  {'Precision':<30}: {self.precision:.4f}  ({self.precision*100:.1f}%)")
        print(f"  {'Recall':<30}: {self.recall:.4f}  ({self.recall*100:.1f}%)")
        print(f"  {'F1 Score':<30}: {self.f1_score:.4f}  ({self.f1_score*100:.1f}%)")
        print(f"  {'Accuracy':<30}: {self.accuracy:.4f}  ({self.accuracy*100:.1f}%)")

        # Interpretation
        print(f"\n  🔍 Interpretation")
        if self.FP > 0:
            print(f"  ⚠️  {self.FP} False Positive(s)  → Cache returned wrong answer (lower threshold)")
        if self.FN > 0:
            print(f"  ⚠️  {self.FN} False Negative(s)  → Cache missed similar question (raise threshold)")
        if self.FP == 0 and self.FN == 0:
            print(f"  🏆 Perfect cache — no false hits or misses!")

        # Latency
        print(f"\n  ⏱️  Latency")
        print(f"  {'Avg Cache Hit Time':<30}: {self.avg_hit_time * 1000:.2f} ms")
        print(f"  {'Avg Cache Miss Time':<30}: {self.avg_miss_time * 1000:.2f} ms")
        print(f"  {'Avg LLM Call Time':<30}: {self.avg_llm_time * 1000:.2f} ms")
        print(f"  {'Est. Time Saved':<30}: {self.total_time_saved:.2f} s")

        # LLM calls made
        print(f"\n  🤖 LLM Calls Made: {len(self.llm_calls)}")
        for i, call in enumerate(self.llm_calls, 1):
            print(f"    [{i}] Q: {call['question'][:55]}...")
            print(f"         A: {call['response'][:55]}...")
        print("=" * 60)


       