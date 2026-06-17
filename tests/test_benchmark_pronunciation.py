"""Tests for the pronunciation benchmark helpers."""

import numpy as np

from scripts.benchmark_pronunciation import (
    default_variants,
    drop_middle_segment,
    drop_tail,
    insert_long_pause,
    local_pitch_shift_down,
    mute_middle_segment,
    noisy,
    pitch_shift_down,
    slow_down,
)


def _tone(sr=16000, duration=1.0):
    t = np.arange(int(sr * duration)) / sr
    return (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_drop_tail_shortens_audio():
    wav = _tone()
    out = drop_tail(wav, ratio=0.25)
    assert 0 < len(out) < len(wav)


def test_drop_middle_segment_shortens_audio():
    wav = _tone(duration=2.0)
    out = drop_middle_segment(wav, duration_seconds=0.25)
    assert 0 < len(out) < len(wav)


def test_mute_middle_segment_keeps_duration_and_adds_silence():
    wav = _tone(duration=2.0)
    out = mute_middle_segment(wav, duration_seconds=0.25)
    assert out.shape == wav.shape
    assert np.count_nonzero(out == 0.0) > np.count_nonzero(wav == 0.0)


def test_insert_long_pause_increases_duration():
    wav = _tone()
    out = insert_long_pause(wav, pause_seconds=0.5)
    assert len(out) == len(wav) + 8000
    assert np.max(np.abs(out)) <= 1.0


def test_slow_down_increases_duration():
    wav = _tone()
    out = slow_down(wav, rate=0.8)
    assert len(out) > len(wav)
    assert np.max(np.abs(out)) <= 1.0


def test_pitch_shift_down_preserves_reasonable_duration():
    wav = _tone()
    out = pitch_shift_down(wav, semitones=-3.0)
    assert abs(len(out) - len(wav)) <= 2
    assert np.max(np.abs(out)) <= 1.0


def test_local_pitch_shift_down_preserves_duration():
    wav = _tone(duration=2.0)
    out = local_pitch_shift_down(wav, semitones=-3.0)
    assert out.shape == wav.shape
    assert np.max(np.abs(out)) <= 1.0
    assert not np.allclose(out, wav)


def test_noisy_keeps_shape_and_normalizes():
    wav = _tone()
    out = noisy(wav, snr_db=12.0)
    assert out.shape == wav.shape
    assert np.max(np.abs(out)) <= 1.0
    assert not np.allclose(out, wav)


def test_default_variants_include_local_error_cases():
    names = {variant.name for variant in default_variants()}
    assert {"drop_middle", "mute_middle", "local_pitch_down"} <= names
