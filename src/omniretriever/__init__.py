"""OmniRetriever: any-to-any audio-video-text retrieval.

Top-level package re-exports the most commonly used entry points so users can do::

    from omniretriever import OmniRetriever, recall_at_k, ndcg_at_10
"""

from omniretriever.models.loader import OmniRetriever
from omniretriever.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_10,
    recall_at_k,
)

__version__ = "0.1.0"

__all__ = [
    "OmniRetriever",
    "recall_at_k",
    "ndcg_at_10",
    "mean_reciprocal_rank",
    "__version__",
]
