#!/usr/bin/env bash
#
# Batch-extract OmniRetriever-7B embeddings for a manifest of records.
#
# Usage:
#   bash scripts/extract_embeddings.sh MANIFEST OUTPUT.npz [-- extra args]
#
# Example (after pulling the benchmark from HF Hub):
#   bash scripts/extract_embeddings.sh \
#       benchmark/manifests/omniretriever_bench.jsonl \
#       output/embeddings/omniretriever_bench.npz
#
# Required env:
#   WAVE_HOME   directory containing WAVE-7B/ and the BEATs checkpoint
#
# Optional env (sensible defaults below):
#   BASE_MODEL  path or HF id of the WAVE-7B backbone
#   ADAPTER     path or HF id of the OmniRetriever LoRA adapter
#   DEVICE      torch device (default: cuda)
#   DTYPE       float32 | bfloat16 | float16 (default: bfloat16)

set -euo pipefail

MANIFEST="${1:?MANIFEST path required}"
OUTPUT="${2:?OUTPUT .npz path required}"
shift 2 || true

BASE_MODEL="${BASE_MODEL:-${WAVE_HOME:?WAVE_HOME or BASE_MODEL must be set}/WAVE-7B}"
ADAPTER="${ADAPTER:-<TBD>/OmniRetriever-7B}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m omniretriever.cli extract \
  "${MANIFEST}" \
  --base-model "${BASE_MODEL}" \
  --adapter   "${ADAPTER}" \
  --output    "${OUTPUT}" \
  --device    "${DEVICE}" \
  --dtype     "${DTYPE}" \
  "$@"
