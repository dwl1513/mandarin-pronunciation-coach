"""Standard pronunciation reference synthesis (M5 TTS).

We expose `synth_reference(text)` which returns a path to a 16 kHz mono WAV
that downstream scoring (especially accuracy DTW) treats as ground truth.

Engines:
    * "edge-tts" — default. Free, fast, uses Microsoft's Azure neural voices
                   via Edge's TTS WebSocket. No GPU, no local model files.
    * "aliyun-tts" — Aliyun DashScope / Qwen-TTS. Useful as a second reference
                     voice for pronunciation-assessment experiments.
    * "mimo-tts" — Xiaomi MiMo-V2.5-TTS. High-quality OpenAI-compatible TTS
                   endpoint with low-latency preset voices.
    * "f5-tts"   — optional heavy local model. Better for "natural example"
                   playback; needs a voice prompt to clone from.

Results are cached on disk so re-running the same reference text doesn't
make a network call every time.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf

from config import (ALIYUN_TTS_LANGUAGE_TYPE, ALIYUN_TTS_MODEL,
                    ALIYUN_TTS_STYLE_PROMPT, ALIYUN_TTS_VOICE,
                    EDGE_TTS_VOICE, MIMO_TTS_BASE_URL, MIMO_TTS_MODEL,
                    MIMO_TTS_STYLE_PROMPT, MIMO_TTS_VOICE, PROJECT_ROOT,
                    SAMPLE_RATE, STD_AUDIO_DIR, TTS_ENGINE)

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


# --------------------------------------------------------------- Aliyun Qwen-TTS
def _load_env_file(path: Optional[str | Path] = None) -> Optional[Path]:
    """Load a simple KEY=VALUE env file without overriding existing env vars."""
    candidates = [
        path,
        os.getenv("MANDARIN_COACH_ENV_FILE"),
        os.getenv("ROS_PLAYGROUND_ENV"),
        PROJECT_ROOT / ".env",
    ]
    for item in candidates:
        if not item:
            continue
        env_path = Path(item).expanduser()
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return env_path
    return None


def _env_value(name: str, default: str) -> str:
    """读取 .env / 环境变量中的配置；已有系统环境变量优先。"""
    _load_env_file()
    value = os.getenv(name, "").strip()
    return value or default


def _dashscope_api_key() -> str:
    _load_env_file()
    for name in (
        "DASHSCOPE_API_KEY",
        "ALIYUN_DASHSCOPE_API_KEY",
        "ALIYUN_API_KEY",
        "BAILIAN_API_KEY",
        "QWEN_API_KEY",
    ):
        api_key = os.getenv(name, "").strip()
        if api_key:
            return api_key
    raise RuntimeError(
        "缺少 DASHSCOPE_API_KEY。可以放在项目 .env 里，或设置 "
        "MANDARIN_COACH_ENV_FILE=/path/to/.env。"
    )


def _dashscope_response_to_dict(response) -> dict:
    if isinstance(response, dict):
        return dict(response)
    if hasattr(response, "to_dict"):
        return dict(response.to_dict())
    try:
        return json.loads(str(response))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DashScope 返回无法解析为 JSON：{response}") from exc


def _extract_aliyun_audio_url(response_data: dict) -> tuple[str, int | None]:
    audio = response_data.get("output", {}).get("audio") or {}
    url = str(audio.get("url") or "").strip()
    expires_at = audio.get("expires_at")
    return url, expires_at if isinstance(expires_at, int) else None


def _download_audio(url: str, output_path: Path, timeout_seconds: float = 60.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        output_path.write_bytes(response.read())


def _assert_dashscope_success(response_data: dict, service_name: str) -> None:
    status_code = response_data.get("status_code")
    if status_code in (None, 200):
        return
    code = response_data.get("code") or ""
    message = response_data.get("message") or ""
    raise RuntimeError(
        f"{service_name} 调用失败：status_code={status_code}, code={code}, message={message}"
    )


def _aliyun_tts(text: str,
                voice: Optional[str] = None,
                model: Optional[str] = None,
                language_type: Optional[str] = None) -> Path:
    """Synthesize via Aliyun DashScope Qwen-TTS → 16 kHz mono WAV path."""
    voice = voice or _env_value("ALIYUN_TTS_VOICE", ALIYUN_TTS_VOICE)
    model = model or _env_value("ALIYUN_TTS_MODEL", ALIYUN_TTS_MODEL)
    language_type = language_type or _env_value(
        "ALIYUN_TTS_LANGUAGE_TYPE", ALIYUN_TTS_LANGUAGE_TYPE,
    )
    style_prompt = _env_value("ALIYUN_TTS_STYLE_PROMPT", ALIYUN_TTS_STYLE_PROMPT)
    cache_key = "|".join([model, voice, language_type, style_prompt])
    cache = _cache_path(text, "aliyun-tts", cache_key)
    if cache.exists():
        return cache

    try:
        import dashscope
    except Exception as e:
        raise RuntimeError("缺少 dashscope 依赖，请执行：uv sync --extra aliyun") from e

    response = dashscope.MultiModalConversation.call(
        api_key=_dashscope_api_key(),
        model=model,
        text=text,
        voice=voice,
        language_type=language_type,
        instructions=style_prompt,
    )
    response_data = _dashscope_response_to_dict(response)
    _assert_dashscope_success(response_data, "阿里云 Qwen-TTS")

    audio_url, _expires_at = _extract_aliyun_audio_url(response_data)
    if not audio_url:
        raise RuntimeError(f"阿里云 Qwen-TTS 没有返回音频 URL：{response_data}")

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "aliyun_tts_audio.wav"
        _download_audio(audio_url, raw_path)
        wav, _ = librosa.load(str(raw_path), sr=SAMPLE_RATE, mono=True)
        sf.write(str(cache), wav, SAMPLE_RATE, subtype="PCM_16")
    return cache


# --------------------------------------------------------------- Mimo TTS
def _mimo_api_key() -> str:
    _load_env_file()
    api_key = os.getenv("MIMO_API_KEY", "").strip()
    if api_key:
        return api_key
    raise RuntimeError("缺少 MIMO_API_KEY。请在项目 .env 里填写。")


def _mimo_tts(text: str,
              voice: Optional[str] = None,
              model: Optional[str] = None) -> Path:
    """Synthesize via Xiaomi MiMo TTS → 16 kHz mono WAV path."""
    voice = voice or _env_value("MIMO_TTS_VOICE", MIMO_TTS_VOICE)
    model = model or _env_value("MIMO_TTS_MODEL", MIMO_TTS_MODEL)
    base_url = _env_value("MIMO_TTS_BASE_URL", MIMO_TTS_BASE_URL)
    style_prompt = _env_value("MIMO_TTS_STYLE_PROMPT", MIMO_TTS_STYLE_PROMPT)

    cache = _cache_path(text, "mimo-tts", "|".join([model, voice, style_prompt]))
    if cache.exists():
        return cache

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("缺少 openai 依赖，请执行：uv sync --extra mimo") from e

    client = OpenAI(api_key=_mimo_api_key(), base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": style_prompt},
            {"role": "assistant", "content": text},
        ],
        audio={"format": "wav", "voice": voice},
    )
    message = completion.choices[0].message
    audio = getattr(message, "audio", None)
    if audio is None:
        raise RuntimeError(f"Mimo TTS 没有返回音频：{completion!r}")
    if isinstance(audio, dict):
        audio_data = audio.get("data")
    else:
        audio_data = getattr(audio, "data", None)
    if not audio_data:
        raise RuntimeError("Mimo TTS 返回里没有 audio.data")

    raw_bytes = base64.b64decode(audio_data)
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "mimo_tts.wav"
        raw_path.write_bytes(raw_bytes)
        wav, _ = librosa.load(str(raw_path), sr=SAMPLE_RATE, mono=True)
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
                    voice: Optional[str] = None,
                    f5_prompt_audio: Optional[Path] = None,
                    f5_prompt_text: Optional[str] = None) -> Path:
    """Synthesize a reference utterance and return its WAV path on disk."""
    engine = engine or _env_value("TTS_ENGINE", TTS_ENGINE)
    if engine == "edge-tts":
        return _edge_tts(text, voice=voice or _env_value("EDGE_TTS_VOICE", EDGE_TTS_VOICE))
    if engine == "aliyun-tts":
        return _aliyun_tts(text, voice=voice)
    if engine == "mimo-tts":
        return _mimo_tts(text, voice=voice)
    if engine == "f5-tts":
        return _f5_tts(text, prompt_audio=f5_prompt_audio,
                       prompt_text=f5_prompt_text)
    raise ValueError(f"Unknown TTS engine: {engine!r}")
