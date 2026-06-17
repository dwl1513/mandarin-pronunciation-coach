"""Audio pre-processing (M1).

Implements the classroom-required DSP front-end **by hand**:
    1. resample to 16 kHz mono            (delegated to librosa)
    2. pre-emphasis filter                y[n] = x[n] - α·x[n-1]
    3. frame blocking + Hamming window    25 ms frames, 10 ms hop
    4. voice-activity detection           webrtcvad (or energy fallback)

The result feeds every downstream module via a single dataclass so that
later code paths can stay agnostic of how the signal was acquired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union
import warnings

import numpy as np

from config import (FRAME_LENGTH, HOP_LENGTH, PRE_EMPHASIS, SAMPLE_RATE,
                    VAD_AGGRESSIVENESS, VAD_FRAME_MS)

from .capture import load_audio

PathLike = Union[str, Path]


# --------------------------------------------------------------------- dataclass
@dataclass
class PreprocessResult:
    """Container shared by every downstream module."""

    wav: np.ndarray                          # post-emphasis, [-1, 1], float32
    raw_wav: np.ndarray                      # original 16k mono, no emphasis
    sr: int                                  # always SAMPLE_RATE (16000)
    frames: np.ndarray                       # [n_frames, FRAME_LENGTH], Hamming
    vad_segments: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.raw_wav) / float(self.sr)

    @property
    def voiced_duration(self) -> float:
        return sum(end - start for start, end in self.vad_segments)


# ------------------------------------------------------------------ DSP helpers
def pre_emphasis(wav: np.ndarray, alpha: float = PRE_EMPHASIS) -> np.ndarray:
    """y[n] = x[n] - α·x[n-1]. Boosts high frequencies to flatten speech tilt."""
    if wav.size == 0:
        return wav
    out = np.empty_like(wav)
    out[0] = wav[0]
    out[1:] = wav[1:] - alpha * wav[:-1]
    return out


def frame_blocking(wav: np.ndarray,
                   frame_length: int = FRAME_LENGTH,
                   hop_length: int = HOP_LENGTH,
                   window: str = "hamming") -> np.ndarray:
    """Split signal into overlapping windowed frames.

    Returns: array of shape [n_frames, frame_length] with the chosen window
    already applied. Last partial frame is zero-padded so callers can rely on
    a rectangular array.
    """
    if wav.size < frame_length:
        # pad so we still produce one frame — keeps downstream code simple
        pad = frame_length - wav.size
        wav = np.pad(wav, (0, pad))

    n_frames = 1 + (len(wav) - frame_length) // hop_length
    # Strided view → contiguous copy avoids aliasing surprises for callers.
    shape = (n_frames, frame_length)
    strides = (wav.strides[0] * hop_length, wav.strides[0])
    frames = np.lib.stride_tricks.as_strided(wav, shape=shape,
                                             strides=strides).copy()

    if window == "hamming":
        win = np.hamming(frame_length).astype(frames.dtype)
    elif window == "hann":
        win = np.hanning(frame_length).astype(frames.dtype)
    elif window in (None, "rect", "none"):
        win = None
    else:
        raise ValueError(f"Unknown window: {window}")

    if win is not None:
        frames = frames * win[np.newaxis, :]
    return frames


# ------------------------------------------------------------------------- VAD
def _vad_webrtc(pcm16: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Run webrtcvad over PCM16 audio, return [(start_s, end_s), ...] of voiced regions."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        import webrtcvad  # local import: optional dep

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_samples = int(sr * VAD_FRAME_MS / 1000)
    bytes_per_frame = frame_samples * 2          # int16 = 2 bytes

    n_frames = len(pcm16) // frame_samples
    voiced_flags = []
    for i in range(n_frames):
        start = i * frame_samples
        frame_bytes = pcm16[start:start + frame_samples].tobytes()
        if len(frame_bytes) != bytes_per_frame:
            voiced_flags.append(False)
            continue
        voiced_flags.append(vad.is_speech(frame_bytes, sr))

    # Merge consecutive voiced frames with short bridging across <=2 silent frames
    segments: List[Tuple[float, float]] = []
    in_speech = False
    seg_start = 0
    silent_run = 0
    bridge = 2                                 # frames of silence we tolerate
    for i, voiced in enumerate(voiced_flags):
        if voiced:
            if not in_speech:
                in_speech = True
                seg_start = i
            silent_run = 0
        else:
            if in_speech:
                silent_run += 1
                if silent_run > bridge:
                    seg_end = i - silent_run + 1
                    segments.append((seg_start * VAD_FRAME_MS / 1000.0,
                                     seg_end * VAD_FRAME_MS / 1000.0))
                    in_speech = False
                    silent_run = 0
    if in_speech:
        segments.append((seg_start * VAD_FRAME_MS / 1000.0,
                         n_frames * VAD_FRAME_MS / 1000.0))

    # Drop spurious < 60 ms blips
    return [s for s in segments if s[1] - s[0] >= 0.06]


def _vad_energy(wav: np.ndarray, sr: int,
                frame_ms: int = 30) -> List[Tuple[float, float]]:
    """Fallback VAD using short-time energy + zero-crossing rate."""
    frame_samples = int(sr * frame_ms / 1000)
    n_frames = len(wav) // frame_samples
    if n_frames == 0:
        return []
    energies = np.empty(n_frames)
    for i in range(n_frames):
        chunk = wav[i * frame_samples:(i + 1) * frame_samples]
        energies[i] = float(np.mean(chunk * chunk))

    # Adaptive threshold: 30 % of mean, but not below noise floor
    thr = max(0.3 * energies.mean(), 1e-5)
    voiced = energies > thr

    segments: List[Tuple[float, float]] = []
    i = 0
    while i < n_frames:
        if voiced[i]:
            j = i
            while j < n_frames and voiced[j]:
                j += 1
            segments.append((i * frame_ms / 1000.0, j * frame_ms / 1000.0))
            i = j
        else:
            i += 1
    return [s for s in segments if s[1] - s[0] >= 0.06]


def detect_voice_segments(wav: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Run webrtcvad on int16 PCM; fall back to energy thresholding on failure."""
    try:
        pcm16 = np.clip(wav, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        return _vad_webrtc(pcm16, sr)
    except Exception:
        return _vad_energy(wav, sr)


# ------------------------------------------------------------------- public API
def preprocess(source: Union[PathLike, np.ndarray, Tuple[int, np.ndarray]],
               sr: int = SAMPLE_RATE,
               *,
               trim_silence: bool = True) -> PreprocessResult:
    """Full M1 pipeline: load → trim → emphasis → frame → VAD.

    Args:
        source:        path / ndarray / (sr, ndarray) tuple (see capture.load_audio).
        sr:            target sample rate, fixed at 16 kHz project-wide.
        trim_silence:  trim leading / trailing silence so we don't penalize
                       speakers who started or stopped slightly late.
    """
    raw = load_audio(source, sr=sr)
    if trim_silence and raw.size > 0:
        # librosa.effects.trim uses top_dB threshold; 35 dB is forgiving.
        import librosa
        raw, _ = librosa.effects.trim(raw, top_db=35)
    if raw.size == 0:
        raw = np.zeros(int(0.1 * sr), dtype=np.float32)

    emphasized = pre_emphasis(raw)
    frames = frame_blocking(emphasized)
    vad_segments = detect_voice_segments(raw, sr)

    return PreprocessResult(
        wav=emphasized,
        raw_wav=raw,
        sr=sr,
        frames=frames,
        vad_segments=vad_segments,
    )
