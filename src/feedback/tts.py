"""Standard pronunciation reference synthesis (M5 TTS).

We expose `synth_reference(text)` which returns a path to a 16 kHz mono WAV
that downstream scoring (especially accuracy DTW) treats as ground truth.

Engines:
    * "edge-tts" — default. Free, fast, uses Microsoft's Azure neural voices
                   via Edge's TTS WebSocket. No GPU, no local model files.
    * "f5-tts"   — heavy local model already installed in the f5tts conda
                   env. Better for "natural example" playback; needs a voice
                   prompt to clone from.

Results are cached on disk so re-running the same reference text doesn't
make a network call every time.
"""
from __future__ import annotations

import asyncio
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from config import CACHE_DIR, EDGE_TTS_VOICE, SAMPLE_RATE, STD_AUDIO_DIR, TTS_ENGINE

_TTS_CACHE = STD_AUDIO_DIR
_TTS_CACHE.mkdir(parents=True, exist_ok=True)


def _cache_path(text: str, engine: str, voice: str) -> Path:
    key = hashlib.md5(f"{engine}|{voice}|{text}".encode("utf-8")).hexdigest()[:16]
    return _TTS_CACHE / f"ref_{engine}_{key}.wav"


# --------------------------------------------------------------- edge-tts
async def _edge_tts_async(text: str, voice: str, out_path: Path) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def _edge_tts(text: str, voice: str = EDGE_TTS_VOICE) -> Path:
    """Synthesize via Microsoft edge-tts → 16 kHz mono WAV path."""
    cache = _cache_path(text, "edge-tts", voice)
    if cache.exists():
        return cache

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = Path(tmp) / "tts.mp3"
        asyncio.run(_edge_tts_async(text, voice, mp3_path))
        # librosa decodes mp3 via audioread/ffmpeg → resample to 16 kHz mono
        wav, _ = librosa.load(str(mp3_path), sr=SAMPLE_RATE, mono=True)
        sf.write(str(cache), wav, SAMPLE_RATE, subtype="PCM_16")
    return cache


# --------------------------------------------------------------- F5-TTS (optional)
def _f5_tts(text: str, prompt_audio: Optional[Path] = None,
            prompt_text: Optional[str] = None) -> Path:
    """Synthesize via F5-TTS. Requires a voice prompt to clone."""
    cache = _cache_path(text, "f5-tts", str(prompt_audio or "default"))
    if cache.exists():
        return cache
    try:
        from f5_tts.api import F5TTS
    except Exception as e:
        raise RuntimeError(f"F5-TTS not available: {e!r}")

    if prompt_audio is None or prompt_text is None:
        raise ValueError("F5-TTS needs a reference audio + reference text prompt.")

    api = F5TTS()
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "tts.wav"
        api.infer(
            ref_file=str(prompt_audio),
            ref_text=prompt_text,
            gen_text=text,
            file_wave=str(wav_path),
            remove_silence=False,
        )
        wav, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        sf.write(str(cache), wav, SAMPLE_RATE, subtype="PCM_16")
    return cache


# --------------------------------------------------------------- public API
def synth_reference(text: str,
                    engine: Optional[str] = None,
                    voice: str = EDGE_TTS_VOICE,
                    f5_prompt_audio: Optional[Path] = None,
                    f5_prompt_text: Optional[str] = None) -> Path:
    """Synthesize a reference utterance and return its WAV path on disk."""
    engine = engine or TTS_ENGINE
    if engine == "edge-tts":
        return _edge_tts(text, voice=voice)
    if engine == "f5-tts":
        return _f5_tts(text, prompt_audio=f5_prompt_audio,
                       prompt_text=f5_prompt_text)
    raise ValueError(f"Unknown TTS engine: {engine!r}")
