"""Shared wav2vec2 model holder.

Loading a 1.2 GB CTC checkpoint is slow, so we cache one instance per process
and reuse it for both `recognize()` (M2 ASR) and `align()` (M2 forced alignment).

We also expose the model's frame rate so callers can convert frame indices to
seconds without hard-coding "20 ms per frame" everywhere.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import DEVICE, HF_CACHE_DIR, SAMPLE_RATE, WAV2VEC2_MODEL_ID


@dataclass
class W2V2Bundle:
    model: object             # transformers.Wav2Vec2ForCTC
    processor: object         # transformers.Wav2Vec2Processor
    device: str               # "cuda" | "cpu"
    blank_id: int             # CTC blank token id
    frame_rate_hz: float      # output frames per second (≈50 for wav2vec2)

    def vocab(self) -> dict:
        return self.processor.tokenizer.get_vocab()


_LOCK = threading.Lock()
_BUNDLE: Optional[W2V2Bundle] = None
_LOAD_ERROR: Optional[BaseException] = None


def _resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_bundle(allow_load: bool = True) -> W2V2Bundle:
    """Lazy-load the wav2vec2 checkpoint exactly once.

    Raises:
        RuntimeError if loading fails — callers that need a graceful fallback
        should catch this.
    """
    global _BUNDLE, _LOAD_ERROR
    if _BUNDLE is not None:
        return _BUNDLE
    if _LOAD_ERROR is not None and not allow_load:
        raise RuntimeError(f"Previous load failed: {_LOAD_ERROR!r}")

    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        try:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            device = _resolve_device()
            processor = Wav2Vec2Processor.from_pretrained(
                WAV2VEC2_MODEL_ID, cache_dir=str(HF_CACHE_DIR),
            )
            # Force safetensors: newer transformers + torch<2.6 refuses
            # `torch.load` on .bin checkpoints (CVE-2025-32434).  The HF
            # safetensors auto-conversion bot ships .safetensors variants
            # for most popular checkpoints including this one.
            try:
                model = Wav2Vec2ForCTC.from_pretrained(
                    WAV2VEC2_MODEL_ID, cache_dir=str(HF_CACHE_DIR),
                    use_safetensors=True,
                ).to(device).eval()
            except Exception:
                # Fallback for older transformers without the kwarg.
                model = Wav2Vec2ForCTC.from_pretrained(
                    WAV2VEC2_MODEL_ID, cache_dir=str(HF_CACHE_DIR),
                ).to(device).eval()

            blank_id = processor.tokenizer.pad_token_id
            if blank_id is None:
                blank_id = 0

            # wav2vec2-base/large strides 320 samples => 50 Hz output rate.
            frame_rate = SAMPLE_RATE / 320.0

            _BUNDLE = W2V2Bundle(
                model=model, processor=processor,
                device=device, blank_id=int(blank_id),
                frame_rate_hz=frame_rate,
            )
            return _BUNDLE
        except BaseException as e:
            _LOAD_ERROR = e
            raise RuntimeError(f"Failed to load wav2vec2 ({WAV2VEC2_MODEL_ID}): {e!r}")


def is_loaded() -> bool:
    return _BUNDLE is not None


def reset_for_tests() -> None:
    """Force re-load on next call (used by integration tests)."""
    global _BUNDLE, _LOAD_ERROR
    _BUNDLE = None
    _LOAD_ERROR = None


def get_log_probs(wav: np.ndarray):
    """Forward a 1-D 16 kHz waveform through the CTC head → log-softmax tensor.

    Returns:
        log_probs : torch.Tensor shape [1, T, V] on the model's device.
        bundle    : the W2V2Bundle (so caller can grab blank_id, frame_rate).
    """
    import torch

    bundle = get_bundle()
    inputs = bundle.processor(
        wav, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False,
    )
    input_values = inputs.input_values.to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(input_values).logits        # [1, T, V]
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs, bundle
