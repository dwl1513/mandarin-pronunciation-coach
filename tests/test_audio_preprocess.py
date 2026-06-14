"""Unit tests for src.audio.preprocess (M1)."""
import numpy as np

from src.audio.preprocess import (frame_blocking, pre_emphasis, preprocess,
                                    detect_voice_segments)


def test_pre_emphasis_first_sample_unchanged(sine_220hz):
    out = pre_emphasis(sine_220hz)
    assert out.shape == sine_220hz.shape
    assert np.isclose(out[0], sine_220hz[0])


def test_pre_emphasis_boosts_high_frequencies(sr):
    # 100 Hz vs 4000 Hz sine: high-freq should keep more energy after emphasis.
    t = np.arange(sr) / sr
    low  = np.sin(2 * np.pi *  100 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 4000 * t).astype(np.float32)
    low_e  = pre_emphasis(low)
    high_e = pre_emphasis(high)
    # Ratio of post/pre RMS — high freq should be amplified more than low.
    assert (high_e.std() / high.std()) > (low_e.std() / low.std())


def test_frame_blocking_shape(sr):
    wav = np.zeros(sr, dtype=np.float32)        # 1 s of audio
    frames = frame_blocking(wav)
    # 25 ms frames, 10 ms hop on 1 s → 1 + (16000 - 400)/160 = 98 frames
    assert frames.shape == (98, 400)


def test_frame_blocking_handles_short_input():
    wav = np.zeros(50, dtype=np.float32)        # shorter than one frame
    frames = frame_blocking(wav)
    assert frames.shape == (1, 400)             # padded to one frame


def test_preprocess_returns_consistent_shapes(speech_like_signal, sr):
    result = preprocess(speech_like_signal, sr=sr)
    assert result.sr == sr
    assert result.wav.ndim == 1
    assert result.frames.ndim == 2
    assert result.frames.shape[1] == 400
    assert result.duration > 0


def test_vad_finds_segments_in_speech(speech_like_signal, sr):
    segs = detect_voice_segments(speech_like_signal, sr)
    assert len(segs) >= 1
    # Each segment should be (start <= end) and within bounds.
    duration = len(speech_like_signal) / sr
    for s, e in segs:
        assert 0.0 <= s < e <= duration + 0.1


def test_vad_finds_no_segments_in_silence(silence, sr):
    segs = detect_voice_segments(silence, sr)
    assert segs == []
