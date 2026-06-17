"""Fundamental frequency extraction (M3 pitch).

We use librosa's pYIN by default. It returns NaNs for unvoiced frames, which
we replace with 0 for downstream code that just wants a curve to draw, and
keep a separate boolean voicing mask for code that needs it (e.g. tone
classification, which must skip unvoiced frames).
"""
from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np

from config import F0_MAX_HZ, F0_MIN_HZ, FRAME_LENGTH, HOP_LENGTH, SAMPLE_RATE


def extract_f0(wav: np.ndarray, sr: int = SAMPLE_RATE,
               method: str = "pyin") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate F0 in Hz over time.

    Returns:
        f0       — shape [T], 0 on unvoiced frames
        times    — shape [T], centre-frame timestamps in seconds
        voiced   — shape [T] bool mask
    """
    if wav.size < FRAME_LENGTH * 2:
        wav = np.pad(wav, (0, FRAME_LENGTH * 2 - wav.size))

    frame_length = FRAME_LENGTH * 2    # pYIN/YIN needs >=2 periods of fmin
    if method == "pyin":
        f0, voiced, _vp = librosa.pyin(
            wav,
            fmin=F0_MIN_HZ, fmax=F0_MAX_HZ,
            sr=sr,
            frame_length=frame_length,
            hop_length=HOP_LENGTH,
        )
    elif method == "yin":
        # YIN 比 pYIN 快很多，适合长篇真人范读批量回归测试。它没有直接
        # 给出 voiced mask，所以用帧能量做一个保守的有声门控。
        f0 = librosa.yin(
            wav,
            fmin=F0_MIN_HZ, fmax=F0_MAX_HZ,
            sr=sr,
            frame_length=frame_length,
            hop_length=HOP_LENGTH,
        )
        rms = librosa.feature.rms(
            y=wav,
            frame_length=frame_length,
            hop_length=HOP_LENGTH,
        )[0]
        n = min(len(f0), len(rms))
        f0 = f0[:n]
        rms = rms[:n]
        energy_floor = max(float(np.percentile(rms, 35)) * 0.8,
                           float(np.max(rms)) * 0.01,
                           1e-5)
        voiced = (rms > energy_floor) & np.isfinite(f0)
    else:
        raise ValueError(f"Unsupported F0 method: {method!r}")
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced = voiced.astype(bool)
    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH).astype(np.float32)
    return f0, times, voiced


def semitone(f0: np.ndarray, ref_hz: float = 100.0) -> np.ndarray:
    """Convert Hz to semitones relative to a reference."""
    safe = np.maximum(f0, 1e-3)
    return 12.0 * np.log2(safe / ref_hz)


def normalize_contour(f0_segment: np.ndarray,
                      voiced_mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalise a voiced F0 segment for shape-only comparison.

    Unvoiced frames (f0==0 or mask==False) are dropped before stats; the
    returned curve only contains voiced samples.
    """
    if voiced_mask is None:
        voiced_mask = f0_segment > 0
    voiced = f0_segment[voiced_mask]
    if voiced.size < 2:
        return np.zeros_like(voiced, dtype=np.float32)
    voiced = 12.0 * np.log2(np.maximum(voiced, 1e-3) / 100.0)   # semitones
    mu = voiced.mean()
    sd = voiced.std()
    if sd < 1e-6:
        return np.zeros_like(voiced, dtype=np.float32)
    return ((voiced - mu) / sd).astype(np.float32)


def resample_contour(contour: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-interpolate a 1-D curve to a fixed length (for template matching)."""
    if contour.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if contour.size == 1:
        return np.full(target_len, contour[0], dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, contour.size)
    tgt_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(tgt_x, src_x, contour).astype(np.float32)
