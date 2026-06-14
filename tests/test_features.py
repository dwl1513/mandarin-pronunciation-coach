"""Unit tests for src.features (M3)."""
import numpy as np

from src.features.pitch import extract_f0, normalize_contour, resample_contour
from src.features.spectral import extract_energy, extract_mfcc, extract_zcr


def test_mfcc_shape(sine_220hz):
    mfcc = extract_mfcc(sine_220hz, add_deltas=True)
    assert mfcc.ndim == 2
    # 13 static + 13 delta + 13 delta2 = 39 dims
    assert mfcc.shape[1] == 39
    assert mfcc.shape[0] > 0


def test_mfcc_handles_empty_input():
    mfcc = extract_mfcc(np.array([], dtype=np.float32))
    assert mfcc.shape[1] in (13, 39)


def test_energy_is_higher_for_louder_signal(sr):
    quiet = 0.01 * np.random.default_rng(0).standard_normal(sr).astype(np.float32)
    loud  = 0.30 * np.random.default_rng(0).standard_normal(sr).astype(np.float32)
    assert extract_energy(loud).mean() > extract_energy(quiet).mean() * 50


def test_zcr_higher_for_high_frequency(sr):
    t = np.arange(sr) / sr
    low  = np.sin(2 * np.pi *  100 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 4000 * t).astype(np.float32)
    assert extract_zcr(high).mean() > extract_zcr(low).mean() * 5


def test_f0_recovers_known_pitch(sr):
    """pYIN on a clean 220 Hz sine should return ~220 Hz on voiced frames."""
    t = np.arange(int(sr * 1.0)) / sr
    sine = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    f0, _times, voiced = extract_f0(sine)
    detected = f0[voiced & (f0 > 0)]
    assert detected.size > 0
    assert abs(np.median(detected) - 220.0) < 10.0


def test_resample_contour_length():
    src = np.linspace(0, 1, 7).astype(np.float32)
    out = resample_contour(src, 20)
    assert out.shape == (20,)
    assert np.isclose(out[0], src[0])
    assert np.isclose(out[-1], src[-1])


def test_normalize_contour_unit_std():
    src = np.array([100, 200, 300, 400], dtype=np.float32)
    z = normalize_contour(src)
    assert z.size == 4
    assert abs(z.std() - 1.0) < 1e-3
