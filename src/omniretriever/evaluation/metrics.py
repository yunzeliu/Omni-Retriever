"""Standard retrieval metrics: Recall@K, NDCG@10, MRR, median rank.

All functions assume a *diagonal-relevance* setup: for an N-by-N similarity
matrix ``S``, sample ``i`` has exactly one relevant gallery item, also
indexed ``i``. This matches the OmniRetriever-Bench and external benchmarks
(MSR-VTT, MSVD, DiDeMo, VATEX, Clotho, SoundDescs) reported in the paper.

Inputs may be ``torch.Tensor`` or ``numpy.ndarray``; the implementation casts
internally and returns Python floats / numpy arrays.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


# --------------------------------------------------------------------------- #
# Core: rank of the gold target for every query                               #
# --------------------------------------------------------------------------- #


def per_query_rank(similarity: np.ndarray) -> np.ndarray:
    """Return the rank (1-indexed) of the diagonal element for every row.

    Args:
        similarity: ``(N, M)`` similarity matrix where row ``i`` is the
            similarity of query ``i`` against every gallery item. The relevant
            target of query ``i`` is gallery item ``i``.

    Returns:
        Integer array of shape ``(N,)`` with ranks in ``[1, M]``.
    """
    sim = _to_numpy(similarity)
    if sim.ndim != 2:
        raise ValueError(f"similarity must be 2-D, got shape {sim.shape}")

    n = sim.shape[0]
    gold_scores = np.diagonal(sim).reshape(-1, 1)
    # Strict-greater + tie-breaking-as-rank ensures ties don't artificially
    # demote the gold target.
    ranks = (sim > gold_scores).sum(axis=1) + 1
    return ranks.astype(np.int64)[:n]


# --------------------------------------------------------------------------- #
# Standard metrics                                                            #
# --------------------------------------------------------------------------- #


def recall_at_k(similarity, k: int | Sequence[int] = (1, 5, 10)) -> dict[int, float]:
    """Compute Recall@K for one or more ``K`` values.

    Args:
        similarity: ``(N, M)`` similarity matrix; the gold target of query
            ``i`` is gallery item ``i``.
        k: a single integer or a sequence of integers.

    Returns:
        Dictionary mapping ``k`` to recall in ``[0, 1]``.
    """
    ks = (k,) if isinstance(k, int) else tuple(k)
    ranks = per_query_rank(similarity)
    out: dict[int, float] = {}
    for kk in ks:
        out[kk] = float(np.mean(ranks <= kk))
    return out


def ndcg_at_10(similarity) -> float:
    """Compute NDCG@10 under the single-relevant-document assumption.

    For diagonal relevance and a single relevant document per query, NDCG@10
    reduces to ``1 / log2(rank + 1)`` if the gold rank is within the top 10,
    otherwise ``0``. The ideal DCG is ``1 / log2(2) = 1``, so no normalisation
    is needed.
    """
    ranks = per_query_rank(similarity)
    in_top10 = ranks <= 10
    dcg = np.where(in_top10, 1.0 / np.log2(ranks + 1), 0.0)
    return float(np.mean(dcg))


def mean_reciprocal_rank(similarity) -> float:
    """Compute mean reciprocal rank (MRR) of the gold target."""
    ranks = per_query_rank(similarity)
    return float(np.mean(1.0 / ranks))


def median_rank(similarity) -> float:
    """Compute the median rank of the gold target (1-indexed, lower is better)."""
    ranks = per_query_rank(similarity)
    return float(np.median(ranks))


# --------------------------------------------------------------------------- #
# Convenience                                                                 #
# --------------------------------------------------------------------------- #


def all_metrics(similarity, k=(1, 5, 10)) -> dict[str, float]:
    """Compute Recall@K, NDCG@10, MRR, median rank in a single pass.

    Returns:
        A dictionary like ``{"R@1": 0.479, "R@5": 0.742, ..., "MRR": 0.602}``
        with all numeric values in ``[0, 1]`` (except median_rank, which is
        in ``[1, M]``).
    """
    out: dict[str, float] = {}
    for kk, v in recall_at_k(similarity, k=k).items():
        out[f"R@{kk}"] = v
    out["NDCG@10"] = ndcg_at_10(similarity)
    out["MRR"] = mean_reciprocal_rank(similarity)
    out["median_rank"] = median_rank(similarity)
    return out
