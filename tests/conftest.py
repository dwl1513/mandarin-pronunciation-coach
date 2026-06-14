"""Shared pytest fixtures + sys.path bootstrap."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------- synthetic-audio fixtures ----------------------------
@pytest.fixture(scope="session")
def sr():
    return 16000


def _sine(freq, duration, sr, amp=0.5):
    t = np.arange(int(sr * duration)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def sine_220hz(sr):
    return _sine(220.0, 1.0, sr)


@pytest.fixture
def silence(sr):
    return np.zeros(int(sr * 0.5), dtype=np.float32)


@pytest.fixture
def speech_like_signal(sr):
    """Synthetic 'speech-like' signal: AM-modulated harmonics with pauses."""
    rng = np.random.default_rng(0)
    chunks = []
    # 4 voiced bursts, each 200 ms, separated by 100 ms silence
    for f0 in (180, 220, 200, 240):
        t = np.arange(int(sr * 0.25)) / sr
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 6 * t)
        sig = sum(
            np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)
        ) * env * 0.4
        sig += rng.normal(0, 0.002, sig.shape)
        chunks.append(sig.astype(np.float32))
        chunks.append(np.zeros(int(sr * 0.1), dtype=np.float32))
    return np.concatenate(chunks)


@pytest.fixture
def f0_contour_factory(sr):
    """Build a synthetic F0 trajectory for a single 'syllable' (200 ms)."""
    def _build(shape: str):
        n = 20  # frames
        if shape == "flat":
            return np.linspace(220, 220, n).astype(np.float32)
        if shape == "rising":
            return np.linspace(180, 260, n).astype(np.float32)
        if shape == "falling":
            return np.linspace(260, 150, n).astype(np.float32)
        if shape == "dipping":
            half = n // 2
            down = np.linspace(220, 140, half)
            up = np.linspace(140, 230, n - half)
            return np.concatenate([down, up]).astype(np.float32)
        raise ValueError(shape)
    return _build
