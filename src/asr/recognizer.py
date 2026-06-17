"""ASR recognition backends for completeness scoring.

Used for:
    1. completeness scoring — how much of the reference did the user actually
       read (character-level edit similarity).
    2. sanity-check — if recognition is wildly off, we can inspect the
       recognized text in the report.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from config import ALIYUN_ASR_MODEL, ASR_ENGINE, PROJECT_ROOT, SAMPLE_RATE

from .models import get_log_probs


def _load_env_file(path: Optional[str | Path] = None) -> Optional[Path]:
    """读取简单 KEY=VALUE 环境文件；已有环境变量优先。"""
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


def _response_to_dict(response) -> dict:
    if isinstance(response, dict):
        return dict(response)
    if hasattr(response, "to_dict"):
        return dict(response.to_dict())
    try:
        return json.loads(str(response))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DashScope 返回无法解析为 JSON：{response}") from exc


def _assert_dashscope_success(response_data: dict, service_name: str) -> None:
    status_code = response_data.get("status_code")
    if status_code in (None, 200):
        return
    code = response_data.get("code") or ""
    message = response_data.get("message") or ""
    raise RuntimeError(
        f"{service_name} 调用失败：status_code={status_code}, code={code}, message={message}"
    )


def _clean_recognized_text(text: str) -> str:
    """只保留 CJK 字符和空白，避免标点/英文 token 干扰完整度。"""
    cleaned = "".join(
        ch for ch in text
        if ("一" <= ch <= "鿿") or ch.isspace()
    )
    return " ".join(cleaned.split())


def _wav_to_data_uri(wav: np.ndarray) -> str:
    """把 16 kHz float32 单声道音频编码成 DashScope 可读的 data URI。"""
    clipped = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, clipped, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _extract_aliyun_asr_text(response_data: dict) -> str:
    choices = response_data.get("output", {}).get("choices") or response_data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                texts.append(str(item["text"]))
        return "".join(texts).strip()

    return ""


def _recognize_wav2vec2(wav: np.ndarray) -> str:
    """Run local wav2vec2 CTC greedy decode on a 16 kHz mono waveform."""
    log_probs, bundle = get_log_probs(wav)
    pred_ids = log_probs.argmax(dim=-1)[0].cpu().tolist()
    raw = bundle.processor.decode(pred_ids).strip()
    return _clean_recognized_text(raw)


def _recognize_aliyun(wav: np.ndarray,
                      model: Optional[str] = None,
                      language: str = "zh",
                      enable_itn: bool = False) -> str:
    """Run Aliyun DashScope Qwen-ASR on a 16 kHz mono waveform."""
    try:
        import dashscope
    except Exception as e:
        raise RuntimeError("缺少 dashscope 依赖，请执行：uv sync --extra aliyun") from e

    model = model or _env_value("ALIYUN_ASR_MODEL", ALIYUN_ASR_MODEL)
    asr_options: dict[str, object] = {"enable_itn": enable_itn}
    if language:
        asr_options["language"] = language

    response = dashscope.MultiModalConversation.call(
        api_key=_dashscope_api_key(),
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{"audio": _wav_to_data_uri(wav)}],
            }
        ],
        result_format="message",
        asr_options=asr_options,
    )
    response_data = _response_to_dict(response)
    _assert_dashscope_success(response_data, "阿里云 Qwen-ASR")
    return _clean_recognized_text(_extract_aliyun_asr_text(response_data))


def recognize(wav: np.ndarray,
              engine: Optional[str] = None,
              *,
              fallback_to_local: bool = True) -> str:
    """Recognize a 16 kHz mono waveform with the configured ASR engine.

    Args:
        wav: 16 kHz mono waveform after preprocessing.
        engine: "wav2vec2", "aliyun-asr", or "auto".
        fallback_to_local: when cloud ASR fails, use local wav2vec2.
    """
    engine = engine or _env_value("ASR_ENGINE", ASR_ENGINE)
    if engine == "wav2vec2":
        return _recognize_wav2vec2(wav)
    if engine == "aliyun-asr":
        try:
            return _recognize_aliyun(wav)
        except Exception:
            if not fallback_to_local:
                raise
            return _recognize_wav2vec2(wav)
    if engine == "auto":
        try:
            return _recognize_aliyun(wav)
        except Exception:
            return _recognize_wav2vec2(wav)
    raise ValueError(f"Unknown ASR engine: {engine!r}")
