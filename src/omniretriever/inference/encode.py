"""Modality-specific encoding routines.

Each ``encode_*`` function takes a (single string or list-of-strings) input,
builds the appropriate multimodal prompt for WAVE-7B, runs a single forward
pass with the all-layer fusion head, and returns L2-normalised embeddings of
shape ``(N, D)`` with ``D = 3584``.

Inputs are batched automatically when a list is provided; users should chunk
manually if memory is tight.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from omniretriever.data.media import load_audio_waveform, load_video_frames

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #


def encode_text(backbone, processor, text, config):
    """Encode a string (or list of strings) into the shared embedding space."""
    texts = _as_list(text)
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_video(backbone, processor, video_path, config):
    """Encode video file(s) using the visual stream only."""
    paths = _as_list(video_path)
    frames = [load_video_frames(p, num_frames=config.video_max_frames,
                                resolution=config.video_resolution) for p in paths]
    prompts = ["<|vision_start|><|video_pad|><|vision_end|>"] * len(paths)
    inputs = processor(
        text=prompts,
        videos=frames,
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_audio(backbone, processor, audio_path, config):
    """Encode audio file(s) using the audio stream only."""
    paths = _as_list(audio_path)
    waveforms = [load_audio_waveform(p,
                                     duration_sec=config.audio_duration_sec,
                                     sample_rate=config.audio_sample_rate) for p in paths]
    prompts = ["<|audio_start|><|audio_pad|><|audio_end|>"] * len(paths)
    inputs = processor(
        text=prompts,
        audios=waveforms,
        sampling_rate=config.audio_sample_rate,
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_av(backbone, processor, clip_path, config):
    """Encode multimodal clip(s) using both visual and audio streams."""
    paths = _as_list(clip_path)
    frames = [load_video_frames(p, num_frames=config.video_max_frames,
                                resolution=config.video_resolution) for p in paths]
    waveforms = [load_audio_waveform(p,
                                     duration_sec=config.audio_duration_sec,
                                     sample_rate=config.audio_sample_rate) for p in paths]
    prompts = [
        "<|vision_start|><|video_pad|><|vision_end|>"
        "<|audio_start|><|audio_pad|><|audio_end|>"
    ] * len(paths)
    inputs = processor(
        text=prompts,
        videos=frames,
        audios=waveforms,
        sampling_rate=config.audio_sample_rate,
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _forward_and_normalise(backbone, inputs, config) -> torch.Tensor:
    """Run a single forward pass and return (optionally L2-normalised) embeddings."""
    outputs = backbone(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    # The all-layer fusion head produces the shared-space embedding as
    # ``outputs.embeds`` (added by the WAVE-7B remote code when
    # ``classify_type=all_layer`` and ``pred_embeds=True`` are active).
    embeddings = getattr(outputs, "embeds", None)
    if embeddings is None:
        raise RuntimeError(
            "Backbone did not return an 'embeds' field. Verify that the LoRA "
            "adapter is applied and that the WAVE-7B remote code is at the "
            "version expected by this release."
        )

    if config.normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    return embeddings


def _as_list(x) -> list:
    if isinstance(x, (str, Path)):
        return [str(x)]
    if isinstance(x, torch.Tensor):
        return [x]
    return [str(item) if isinstance(item, Path) else item for item in x]


def _device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device
