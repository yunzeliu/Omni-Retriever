"""Thin wrapper around the WAVE-7B backbone (Qwen2.5-Omni + BEATs).

The WAVE-7B backbone is a third-party model; its full source lives under the
upstream repository. This module provides a single ``load_wave_backbone``
helper that returns a ``transformers``-style model ready for LoRA injection.

For the released camera-ready, the implementation expects the upstream WAVE-7B
repository to be either (a) installed as a pip package, (b) cloned next to this
package, or (c) discoverable via the ``WAVE_HOME`` environment variable. See
``docs/installation.md`` for setup details.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_wave_backbone(
    model_path: str | os.PathLike,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    use_beats: bool = True,
    beats_checkpoint: str | os.PathLike | None = None,
):
    """Load the WAVE-7B backbone with the BEATs audio adaptor attached.

    Args:
        model_path: path to the WAVE-7B model directory (Hugging Face layout).
        torch_dtype: parameter precision (default ``torch.bfloat16``).
        use_beats: whether to attach the BEATs audio encoder (default ``True``;
            required for audio-anchored retrieval).
        beats_checkpoint: optional override for the BEATs audio-encoder
            checkpoint. If ``None``, the loader looks for
            ``BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`` under
            ``WAVE_HOME`` (set in the environment).

    Returns:
        A WAVE-7B ``PreTrainedModel`` ready to have a LoRA adapter applied.
    """
    wave_home = Path(os.environ.get("WAVE_HOME", "."))

    # ``transformers.AutoModel.from_pretrained`` is sufficient for the
    # underlying Qwen2.5-Omni architecture. Custom branches (BEATs, all-layer
    # fusion head) live in WAVE's own model file and are loaded with
    # ``trust_remote_code=True``.
    AutoModel = importlib.import_module("transformers").AutoModel

    logger.info("Loading WAVE-7B from %s (dtype=%s)", model_path, torch_dtype)
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if use_beats:
        ckpt = beats_checkpoint or (wave_home / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"BEATs audio-encoder checkpoint not found at {ckpt}. "
                "Set WAVE_HOME to the directory containing the BEATs .pt file "
                "or pass beats_checkpoint explicitly."
            )
        _attach_beats(model, ckpt)

    return model


def _attach_beats(model, beats_ckpt_path: Path) -> None:
    """Attach the BEATs audio encoder to a WAVE-7B model.

    BEATs is loaded as a frozen submodule; the trainable LayerNorm and projector
    that map BEATs hidden states into the LLM token stream live inside the
    LoRA adapter (modules_to_save = ``beats_ln``, ``beats_proj``).
    """
    # The actual BEATs loading code lives in the WAVE source tree under
    # ``model/qwen2_5_omni/beats/``. We delegate to whatever utility WAVE
    # exposes (it sets up the encoder, freezes its parameters, and wires the
    # adaptor modules onto the model).
    try:
        from omniretriever.models.beats_adaptor import attach_beats
    except ImportError as exc:
        raise ImportError(
            "The BEATs adaptor module is not bundled with this release. "
            "Clone the WAVE-7B upstream repository and copy "
            "model/qwen2_5_omni/beats/ into omniretriever/models/."
        ) from exc

    attach_beats(model, beats_ckpt_path)
