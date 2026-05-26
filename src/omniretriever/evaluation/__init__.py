"""Evaluation utilities: retrieval metrics and benchmark scoring."""

from omniretriever.evaluation.metrics import (
    mean_reciprocal_rank,
    median_rank,
    ndcg_at_10,
    recall_at_k,
)
from omniretriever.evaluation.score import score_benchmark

__all__ = [
    "recall_at_k",
    "ndcg_at_10",
    "mean_reciprocal_rank",
    "median_rank",
    "score_benchmark",
]
