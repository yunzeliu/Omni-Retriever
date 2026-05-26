"""Lightweight media loaders shared between training and inference.

The loaders intentionally do not import heavy frameworks at module-import time;
all heavy dependencies (``decord``, ``librosa``) are imported lazily inside the
function bodies so users who only need text encoding can avoid the cost.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Video                                                                       #
# --------------------------------------------------------------------------- #


def load_video_frames(
    path: str | os.PathLike,
    *,
    num_frames: int = 8,
    resolution: int = 224,
    sampling: str = "uniform",
) -> np.ndarray:
    """Load ``num_frames`` frames from a video as a ``(N, H, W, 3)`` ``uint8`` array.

    Args:
        path: path to the input video file.
        num_frames: number of frames to sample.
        resolution: target square resolution for each frame.
        sampling: ``"uniform"`` (default, evenly spaced) or ``"random"`` for a
            random subset of ``num_frames`` frames.

    Returns:
        A NumPy array of shape ``(num_frames, resolution, resolution, 3)``,
        ``dtype=uint8``, RGB.
    """
    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("native")

    path = str(path)
    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(reader)
    if total == 0:
        raise RuntimeError(f"Video {path!r} has zero frames.")

    if sampling == "uniform":
        idx = np.linspace(0, max(total - 1, 0), num=num_frames, dtype=int)
    elif sampling == "random":
        idx = np.sort(
            np.random.default_rng().choice(total, size=min(num_frames, total), replace=False)
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling!r}")

    frames = reader.get_batch(idx).asnumpy()  # (N, H, W, 3) uint8
    frames = _resize_frames(frames, resolution)
    return frames


def _resize_frames(frames: np.ndarray, resolution: int) -> np.ndarray:
    """Centre-crop and resize a batch of frames to ``(resolution, resolution)``."""
    from PIL import Image

    out = np.empty((frames.shape[0], resolution, resolution, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize(
            (resolution, resolution), Image.BICUBIC
        )
        out[i] = np.asarray(img)
    return out


# --------------------------------------------------------------------------- #
# Audio                                                                       #
# --------------------------------------------------------------------------- #


def load_audio_waveform(
    path: str | os.PathLike,
    *,
    duration_sec: int = 8,
    sample_rate: int = 16_000,
    pad_mode: str = "zero",
) -> np.ndarray:
    """Load a fixed-duration mono waveform from a video or audio container.

    Args:
        path: path to the source file. Video containers (``.mp4``, ``.mov``,
            ``.webm``) are decoded for their audio track; pure audio files
            (``.wav``, ``.flac``, ``.mp3``) are loaded directly.
        duration_sec: target duration in seconds. Longer waveforms are centre-
            cropped, shorter ones are padded (see ``pad_mode``).
        sample_rate: target sample rate (Hz).
        pad_mode: how to pad short clips. One of ``"zero"`` (silence) or
            ``"loop"`` (repeat the source until it fills the window).

    Returns:
        A NumPy array of shape ``(duration_sec * sample_rate,)``, ``dtype=float32``.
    """
    import librosa

    path = str(path)
    target_len = int(duration_sec * sample_rate)

    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.shape[0] >= target_len:
        # Centre crop.
        start = (waveform.shape[0] - target_len) // 2
        return waveform[start:start + target_len]

    # Pad.
    if pad_mode == "zero":
        out = np.zeros(target_len, dtype=np.float32)
        out[: waveform.shape[0]] = waveform
        return out
    if pad_mode == "loop":
        n = (target_len // max(waveform.shape[0], 1)) + 1
        return np.tile(waveform, n)[:target_len]

    raise ValueError(f"Unknown pad_mode: {pad_mode!r}")
