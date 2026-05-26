#!/usr/bin/env bash
#
# Score paired embeddings against the OmniRetriever-Bench setup.
#
# Usage:
#   bash scripts/evaluate.sh EMBEDDINGS_PATH
#
# EMBEDDINGS_PATH may be:
#   * a single .bin / .npz file with "text_embeds" and "mllm_embeds" arrays, or
#   * a directory of such files (one per task).
#
# Output: JSON with Recall@1/5/10, NDCG@10, MRR, median rank per direction.

set -euo pipefail

EMBEDDINGS="${1:?EMBEDDINGS_PATH required}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m omniretriever.cli evaluate "${EMBEDDINGS}" "${@:2}"
