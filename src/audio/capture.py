"""Audio IO helpers.

The Gradio UI handles microphone capture, so this module focuses on
loading a file path / NumPy array into a normalized 16 kHz mono signal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from config import SAMPLE_RATE

PathLike = Union[str, Path]


def load_audio(source: Union[PathLike, np.ndarray, Tuple[int, np.ndarray]],
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio from a path / array / (sr, array) tuple into mono float32 @ sr.

    Accepts the shapes Gradio sends back from `gr.Audio`:
        - file path (str / Path)
        - (sample_rate, np.ndarray) tuple
        - bare np.ndarray (already at target sr)
    """
    if isinstance(source, (str, Path)):
        wav, file_sr = sf.read(str(source), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        return wav.astype(np.float32, copy=False)

    if isinstance(source, tuple) and len(source) == 2:
        file_sr, arr = source
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        # Gradio commonly hands back int16 PCM — normalize to [-1, 1] float32.
        if np.issubdtype(arr.dtype, np.integer):
            max_val = float(np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32) / max_val
        else:
            arr = arr.astype(np.float32, copy=False)
        if file_sr != sr:
            arr = librosa.resample(arr, orig_sr=file_sr, target_sr=sr)
        return arr

    if isinstance(source, np.ndarray):
        return source.astype(np.float32, copy=False)

    raise TypeError(f"Unsupported audio source type: {type(source)!r}")


def save_audio(wav: np.ndarray, path: PathLike, sr: int = SAMPLE_RATE) -> Path:
    """Write a float32 mono signal to disk as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="PCM_16")
    return path
