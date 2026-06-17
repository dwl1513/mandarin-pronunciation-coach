"""Benchmark pronunciation scorers with controlled synthetic mistakes.

This script builds a small experiment table for reports / slides:

1. synthesize a clean reference reading with the configured TTS;
2. create several controlled variants, such as dropped ending, missing middle
   syllable, local silence, long pause, slower reading, and pitch-shifted
   reading;
3. run the full assessment pipeline on every variant;
4. write Markdown + CSV tables under data/cache/benchmarks/.

The goal is not to replace a real human-labeled corpus.  It shows that each
score dimension responds to the kind of error it is supposed to measure.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import numpy as np  # noqa: E402

from config import CACHE_DIR, SAMPLE_RATE  # noqa: E402
from src.audio.capture import load_audio, save_audio  # noqa: E402
from src.feedback.tts import synth_reference  # noqa: E402
from src.pipeline import assess  # noqa: E402


DEFAULT_TEXT = "今天天气真好，我们一起去公园散步吧。"


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    transform: Callable[[np.ndarray, int], np.ndarray]


def _normalize(wav: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 1e-6:
        return wav.astype(np.float32)
    return (0.95 * wav / peak).astype(np.float32)


def _fit_length(segment: np.ndarray, target_len: int) -> np.ndarray:
    """把局部变换后的片段裁剪或补零到原长度。"""
    if len(segment) > target_len:
        return segment[:target_len].astype(np.float32)
    if len(segment) < target_len:
        pad = np.zeros(target_len - len(segment), dtype=np.float32)
        return np.concatenate([segment.astype(np.float32), pad])
    return segment.astype(np.float32)


def _region_indices(wav: np.ndarray,
                    sr: int,
                    center_ratio: float,
                    duration_seconds: float) -> tuple[int, int]:
    """按比例取一个局部片段，用来模拟某个字附近的错误。"""
    if wav.size == 0:
        return 0, 0
    center = int(len(wav) * center_ratio)
    half = max(1, int(sr * duration_seconds / 2))
    start = max(0, center - half)
    end = min(len(wav), center + half)
    return start, max(start + 1, end)


def identity(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    return wav.astype(np.float32, copy=True)


def drop_tail(wav: np.ndarray, sr: int = SAMPLE_RATE, ratio: float = 0.20) -> np.ndarray:
    """模拟漏读句尾。"""
    keep = max(1, int(len(wav) * (1.0 - ratio)))
    return wav[:keep].astype(np.float32, copy=True)


def drop_middle_segment(wav: np.ndarray,
                        sr: int = SAMPLE_RATE,
                        center_ratio: float = 0.55,
                        duration_seconds: float = 0.35) -> np.ndarray:
    """模拟中间漏读一两个字，音频总时长会变短。"""
    start, end = _region_indices(wav, sr, center_ratio, duration_seconds)
    return np.concatenate([wav[:start], wav[end:]]).astype(np.float32)


def mute_middle_segment(wav: np.ndarray,
                        sr: int = SAMPLE_RATE,
                        center_ratio: float = 0.55,
                        duration_seconds: float = 0.35) -> np.ndarray:
    """模拟某个字含混到接近无声，保留整体节奏位置。"""
    start, end = _region_indices(wav, sr, center_ratio, duration_seconds)
    out = wav.astype(np.float32, copy=True)
    out[start:end] = 0.0
    return out


def insert_long_pause(wav: np.ndarray, sr: int = SAMPLE_RATE,
                      at_ratio: float = 0.50,
                      pause_seconds: float = 1.2) -> np.ndarray:
    """模拟中间卡顿。"""
    idx = int(len(wav) * at_ratio)
    pause = np.zeros(int(sr * pause_seconds), dtype=np.float32)
    return np.concatenate([wav[:idx], pause, wav[idx:]]).astype(np.float32)


def slow_down(wav: np.ndarray, sr: int = SAMPLE_RATE, rate: float = 0.72) -> np.ndarray:
    """模拟整体语速过慢。"""
    stretched = librosa.effects.time_stretch(wav.astype(np.float32), rate=rate)
    return _normalize(stretched)


def pitch_shift_down(wav: np.ndarray, sr: int = SAMPLE_RATE,
                     semitones: float = -4.0) -> np.ndarray:
    """模拟整体音高走势偏离标准音。"""
    shifted = librosa.effects.pitch_shift(
        wav.astype(np.float32), sr=sr, n_steps=semitones,
    )
    return _normalize(shifted)


def local_pitch_shift_down(wav: np.ndarray,
                           sr: int = SAMPLE_RATE,
                           center_ratio: float = 0.55,
                           duration_seconds: float = 0.45,
                           semitones: float = -5.0) -> np.ndarray:
    """模拟局部声调走势错误，只改变中间某个片段的音高。"""
    start, end = _region_indices(wav, sr, center_ratio, duration_seconds)
    out = wav.astype(np.float32, copy=True)
    segment = out[start:end]
    shifted = librosa.effects.pitch_shift(segment, sr=sr, n_steps=semitones)
    out[start:end] = _fit_length(shifted, end - start)
    return _normalize(out)


def noisy(wav: np.ndarray, sr: int = SAMPLE_RATE, snr_db: float = 15.0) -> np.ndarray:
    """模拟轻度背景噪声。"""
    rng = np.random.default_rng(0)
    signal_power = float(np.mean(wav * wav))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return _normalize(wav + noise)


def default_variants() -> list[Variant]:
    return [
        Variant("clean", "标准音原样输入，作为健康链路上限", identity),
        Variant("drop_tail", "删除句尾 20%，模拟漏读", drop_tail),
        Variant("drop_middle", "删除中间约 0.35 秒，模拟中间漏读", drop_middle_segment),
        Variant("mute_middle", "中间约 0.35 秒置静音，模拟某个字发音缺失", mute_middle_segment),
        Variant("long_pause", "中间插入 1.2 秒静音，模拟卡顿", insert_long_pause),
        Variant("slow", "整体拉慢到 72%，模拟语速偏慢", slow_down),
        Variant("pitch_down", "整体降低 4 个半音，模拟音高/声调走势偏离", pitch_shift_down),
        Variant("local_pitch_down", "中间局部降低 5 个半音，模拟局部声调错误", local_pitch_shift_down),
        Variant("noise", "加入 15 dB SNR 背景噪声，模拟录音环境干扰", noisy),
    ]


def _score_variant(audio_path: Path, text: str,
                   tts_engine: str,
                   asr_engine: str) -> dict:
    art = assess(
        audio_path,
        text,
        use_asr=True,
        use_tts_reference=True,
        asr_engine=asr_engine,
        tts_engine=tts_engine,
    )
    dims = art.report["dims"]
    return {
        "overall": art.report["overall"],
        "accuracy": dims.get("accuracy", 0.0),
        "tone": dims.get("tone", 0.0),
        "fluency": dims.get("fluency", 0.0),
        "prosody": dims.get("prosody", 0.0),
        "completeness": dims.get("completeness", 0.0),
        "recognized": art.recognized_text,
    }


def run_benchmark(text: str = DEFAULT_TEXT,
                  tts_engine: str = "mimo-tts,aliyun-tts",
                  asr_engine: str = "aliyun-asr",
                  output_dir: Path = CACHE_DIR / "benchmarks",
                  variants: list[Variant] | None = None) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = variants or default_variants()

    primary_engine = tts_engine.split(",", 1)[0].strip()
    reference_path = synth_reference(text, engine=primary_engine)
    base_wav = load_audio(reference_path)

    rows: list[dict] = []
    for variant in variants:
        wav = variant.transform(base_wav, SAMPLE_RATE)
        audio_path = save_audio(wav, output_dir / f"{variant.name}.wav")
        print(f"[benchmark] {variant.name}: {variant.description}")
        row = {
            "variant": variant.name,
            "description": variant.description,
            "audio_path": str(audio_path),
            **_score_variant(audio_path, text, tts_engine, asr_engine),
        }
        rows.append(row)

    write_outputs(rows, output_dir)
    return rows


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    csv_path = output_dir / "pronunciation_benchmark.csv"
    md_path = output_dir / "pronunciation_benchmark.md"
    fields = [
        "variant", "description", "overall", "accuracy", "tone",
        "fluency", "prosody", "completeness", "recognized", "audio_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 发音评测错误模拟实验",
        "",
        "| 样本 | 说明 | 总分 | 准确度 | 声调 | 流利度 | 韵律 | 完整度 | 识别结果 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['description']} | "
            f"{row['overall']:.1f} | {row['accuracy']:.1f} | {row['tone']:.1f} | "
            f"{row['fluency']:.1f} | {row['prosody']:.1f} | "
            f"{row['completeness']:.1f} | {row['recognized']} |"
        )
    lines.extend([
        "",
        f"CSV: `{csv_path}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[benchmark] CSV written to {csv_path}")
    print(f"[benchmark] Markdown written to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="普通话发音评测错误模拟实验")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help="用于生成标准音和模拟错误的文本")
    parser.add_argument("--tts-engine", default="mimo-tts,aliyun-tts",
                        help="参考音 TTS 引擎，支持逗号分隔多参考音")
    parser.add_argument("--asr-engine", default="aliyun-asr",
                        choices=["wav2vec2", "aliyun-asr", "auto"],
                        help="完整度识别 ASR 引擎")
    parser.add_argument("--output-dir", type=Path,
                        default=CACHE_DIR / "benchmarks",
                        help="输出 wav / CSV / Markdown 的目录")
    args = parser.parse_args()

    run_benchmark(
        text=args.text,
        tts_engine=args.tts_engine,
        asr_engine=args.asr_engine,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
