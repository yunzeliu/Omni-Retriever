"""Command-line interface entry points.

Two sub-commands are exposed::

    python -m omniretriever.cli extract  [args ...]
    python -m omniretriever.cli evaluate [args ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("omniretriever")


def extract_main(argv: list[str] | None = None) -> int:
    """Entry point for ``omniretriever-extract``.

    The manifest is a JSON or JSONL file with one record per line/element::

        {"id": "...", "text": "...", "video": "videos/clip.mp4", "audio": "videos/clip.wav"}

    Any modality field may be missing; the embedding-extraction routine routes
    each record through the matching ``encode_*`` method.
    """
    parser = _build_extract_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from omniretriever import OmniRetriever

    model = OmniRetriever.from_pretrained(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        dtype=args.dtype,
    )

    records = _load_manifest(args.manifest)
    out: dict[str, np.ndarray] = {}

    for record in records:
        record_id = record["id"]
        if "text" in record:
            out[f"{record_id}__text"] = model.encode_text(record["text"]).cpu().numpy()
        if "video" in record and "audio" in record:
            out[f"{record_id}__av"] = model.encode_av(record["video"]).cpu().numpy()
        elif "video" in record:
            out[f"{record_id}__video"] = model.encode_video(record["video"]).cpu().numpy()
        elif "audio" in record:
            out[f"{record_id}__audio"] = model.encode_audio(record["audio"]).cpu().numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **out)
    logger.info("Wrote %d embeddings to %s", len(out), output_path)
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    """Entry point for ``omniretriever-evaluate``."""
    parser = _build_evaluate_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from omniretriever.evaluation.score import score_benchmark, score_directory

    src = Path(args.embeddings)
    if src.is_dir():
        results = score_directory(src)
    else:
        results = {src.stem: score_benchmark(src)}

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _build_extract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omniretriever-extract",
        description="Extract OmniRetriever-7B embeddings for a manifest of records.",
    )
    parser.add_argument("manifest", help="Path to a JSON / JSONL manifest.")
    parser.add_argument("--base-model", required=True, help="Path to the WAVE-7B backbone.")
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output", required=True, help="Output .npz file.")
    parser.add_argument("--device", default="cuda", help="Torch device (default cuda).")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
        help="Inference precision (default bfloat16).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def _build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omniretriever-evaluate",
        description="Score paired embeddings against the OmniRetriever-Bench setup.",
    )
    parser.add_argument(
        "embeddings",
        help="Either a single .bin / .npz file or a directory of them.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def _load_manifest(path: str) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    blob = json.loads(text)
    if isinstance(blob, list):
        return blob
    raise ValueError(f"Manifest must be a list (.json) or JSONL; got {type(blob).__name__}.")


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING - 10 * min(verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch to ``extract`` or ``evaluate`` based on the first argument."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: python -m omniretriever.cli {extract,evaluate} [args ...]",
            file=sys.stderr,
        )
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "extract":
        return extract_main(rest)
    if cmd == "evaluate":
        return evaluate_main(rest)
    print(f"unknown sub-command {cmd!r}; expected 'extract' or 'evaluate'.", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
