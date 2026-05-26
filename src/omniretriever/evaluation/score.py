"""Benchmark scoring: load embeddings, build similarity, report metrics.

This module implements the canonical scoring pipeline used in the paper.
Embeddings are read from ``.npy`` or ``.bin`` files (the latter being the
``torch.save`` dictionary format produced by the training pipeline), the
similarity matrix is computed with cosine similarity (assuming L2-normalised
embeddings), and Recall@K / NDCG@10 are reported per direction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from omniretriever.evaluation.metrics import all_metrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Embedding I/O                                                               #
# --------------------------------------------------------------------------- #


def load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from ``.npy`` or ``.bin`` (torch dict) format.

    ``.bin`` files are expected to contain a dict with key ``"embeddings"``
    (or ``"text_embeds"`` / ``"mllm_embeds"`` for paired text/mm formats);
    callers should specify which subset to use via :func:`load_paired_embeddings`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if suffix == ".bin":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        if "embeddings" not in blob:
            raise KeyError(
                f"{path} does not contain an 'embeddings' key; "
                f"available keys: {sorted(blob)}"
            )
        return np.asarray(blob["embeddings"], dtype=np.float32)
    raise ValueError(f"Unsupported embedding format: {suffix!r} (expected .npy or .bin)")


def load_paired_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load paired (text, multimodal) embeddings from a single ``.bin`` file.

    Returns:
        Tuple ``(text_embeds, mllm_embeds)`` of shape ``(N, D)`` each.
    """
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    text = np.asarray(blob["text_embeds"], dtype=np.float32)
    mm = np.asarray(blob["mllm_embeds"], dtype=np.float32)
    if text.shape != mm.shape:
        raise ValueError(
            f"Paired text and multimodal embeddings must have matching shapes; "
            f"got text={text.shape}, multimodal={mm.shape}"
        )
    return text, mm


# --------------------------------------------------------------------------- #
# Similarity                                                                  #
# --------------------------------------------------------------------------- #


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every pair of rows in ``a`` and ``b``.

    Inputs are normalised first to be safe even if upstream forgot.
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a @ b.T


# --------------------------------------------------------------------------- #
# Top-level scoring                                                           #
# --------------------------------------------------------------------------- #


def score_benchmark(
    embeddings_path: str | Path,
    *,
    directions: Sequence[str] = ("t2m", "m2t"),
) -> Mapping[str, Mapping[str, float]]:
    """Score a single (text, multimodal) embedding pair file.

    Args:
        embeddings_path: path to a ``.bin`` file containing ``text_embeds``
            and ``mllm_embeds`` arrays of shape ``(N, D)``.
        directions: which directions to evaluate. ``"t2m"`` is text-to-mm
            retrieval (text query, mm gallery); ``"m2t"`` is the reverse.

    Returns:
        Nested dictionary mapping each direction to a metrics dictionary
        (R@1, R@5, R@10, NDCG@10, MRR, median_rank).
    """
    text, mm = load_paired_embeddings(embeddings_path)
    sim = cosine_similarity(text, mm)

    out: dict[str, dict[str, float]] = {}
    if "t2m" in directions:
        out["t2m"] = all_metrics(sim)
    if "m2t" in directions:
        out["m2t"] = all_metrics(sim.T)

    return out


def score_directory(
    directory: str | Path,
    *,
    pattern: str = "*.bin",
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    """Score every embedding bin under ``directory``.

    Useful for reproducing the multi-task results table::

        results = score_directory("output/embeddings/")
        for task, dirs in results.items():
            print(task, dirs["t2m"]["R@1"], dirs["m2t"]["R@1"])
    """
    directory = Path(directory)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted(directory.glob(pattern)):
        task = path.stem
        logger.info("Scoring %s", task)
        out[task] = score_benchmark(path)
    return out
