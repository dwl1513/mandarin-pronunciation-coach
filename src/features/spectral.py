"""Spectral features (M3): MFCC, FBank, short-time energy, ZCR.

These are the bread-and-butter speech features. We keep our own thin wrappers
around librosa so that the rest of the codebase imports through a stable
project API rather than touching librosa directly.
"""
from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np

from config import (HOP_LENGTH, FRAME_LENGTH, N_FFT, N_MELS, N_MFCC,
                    SAMPLE_RATE)


def extract_mfcc(wav: np.ndarray,
                 sr: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC,
                 add_deltas: bool = True) -> np.ndarray:
    """Compute MFCC[+Δ+ΔΔ] of shape [T, dim].

    With `add_deltas=True` the returned matrix concatenates the static
    coefficients with their 1st- and 2nd-order time derivatives — the standard
    39-dim feature set used by classical ASR (and a good DTW substrate).
    """
    if wav.size == 0:
        wav = np.zeros(FRAME_LENGTH, dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=wav, sr=sr,
        n_mfcc=n_mfcc, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=FRAME_LENGTH,
        n_mels=N_MELS,
    )
    feats = [mfcc]
    if add_deltas:
        try:
            feats.append(librosa.feature.delta(mfcc, order=1))
            feats.append(librosa.feature.delta(mfcc, order=2))
        except librosa.util.exceptions.ParameterError:
            # Too few frames for delta computation — skip silently.
            pass
    return np.concatenate(feats, axis=0).T.astype(np.float32)


def extract_fbank(wav: np.ndarray, sr: int = SAMPLE_RATE,
                  n_mels: int = N_MELS) -> np.ndarray:
    """log-Mel filterbank, shape [T, n_mels]."""
    if wav.size == 0:
        wav = np.zeros(FRAME_LENGTH, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=FRAME_LENGTH,
        n_mels=n_mels,
    )
    return librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)


def extract_energy(wav: np.ndarray,
                   frame_length: int = FRAME_LENGTH,
                   hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Short-time energy curve, one value per frame."""
    if wav.size < frame_length:
        wav = np.pad(wav, (0, frame_length - wav.size))
    n_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = wav[i * hop_length:i * hop_length + frame_length]
        energy[i] = float(np.sum(chunk * chunk))
    return energy


def extract_zcr(wav: np.ndarray,
                frame_length: int = FRAME_LENGTH,
                hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Zero-crossing rate per frame."""
    return librosa.feature.zero_crossing_rate(
        y=wav, frame_length=frame_length, hop_length=hop_length,
    )[0].astype(np.float32)


def stft_magnitude(wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrogram, returned as (|S|, freqs)."""
    S = np.abs(librosa.stft(wav, n_fft=N_FFT,
                            hop_length=HOP_LENGTH,
                            win_length=FRAME_LENGTH))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    return S.astype(np.float32), freqs.astype(np.float32)
