"""从真人 PSC 范读数据集生成前端正式例句和标准音片段。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/psc_human_dataset/manifest.json"
SPECS_PATH = PROJECT_ROOT / "data/psc_human_dataset/example_clip_specs.json"
CLIPS_PATH = PROJECT_ROOT / "data/psc_human_dataset/example_clips.json"
EXAMPLES_JSON = PROJECT_ROOT / "frontend/src/data/practice-examples.json"
AUDIO_OUTPUT_DIR = PROJECT_ROOT / "frontend/public/audio/examples"


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = {int(item["number"]): item for item in manifest["entries"]}
    specs = json.loads(SPECS_PATH.read_text(encoding="utf-8"))
    manual_clips = _load_manual_clips()

    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    for spec in specs:
        entry = entries[int(spec["number"])]
        text = _take_sentences(str(entry["text"]), int(spec["sentences"]))
        audio_path = _bbmu_audio_path(entry)
        output_path = AUDIO_OUTPUT_DIR / f"{spec['id']}.wav"
        manual_clip = manual_clips.get(str(spec["id"]))
        duration = _write_reference_clip(
            audio_path=audio_path,
            output_path=output_path,
            clip_text=text,
            full_syllable_count=int(entry["syllable_count"]),
            manual_clip=manual_clip,
        )
        examples.append({
            "id": spec["id"],
            "title": f"{spec['title']} · PSC 片段",
            "level": spec["level"],
            "focus": spec["focus"],
            "scene": "普通话水平测试朗读",
            "duration": _format_duration(duration),
            "text": text,
            "tags": spec["tags"],
            "audioPath": f"/audio/examples/{spec['id']}.wav",
        })
        print(f"生成 {spec['id']}：{duration:.2f}s | {text}")

    EXAMPLES_JSON.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"写入例句库：{EXAMPLES_JSON.relative_to(PROJECT_ROOT)}")


def _load_manual_clips() -> dict[str, dict]:
    if not CLIPS_PATH.exists():
        return {}
    data = json.loads(CLIPS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    return {str(item["id"]): item for item in data if isinstance(item, dict) and "id" in item}


def _take_sentences(text: str, count: int) -> str:
    cleaned = re.sub(r"\s+", "", text)
    cleaned = cleaned.replace("〞", "”")
    parts = [part for part in re.split(r"(?<=[。！？])", cleaned) if part]
    return "".join(parts[:count]).strip()


def _bbmu_audio_path(entry: dict) -> Path:
    for audio in entry["audios"]:
        if audio["source_id"] == "bbmu":
            return PROJECT_ROOT / audio["wav_path"]
    raise ValueError(f"缺少 bbmu 真人范读：{entry['title']}")


def _write_reference_clip(
    audio_path: Path,
    output_path: Path,
    clip_text: str,
    full_syllable_count: int,
    manual_clip: dict | None = None,
) -> float:
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if manual_clip:
        start = max(0.0, float(manual_clip["start"]))
        end = max(start + 0.2, float(manual_clip["end"]))
    else:
        speech_start, speech_end = _speech_bounds(wav, sr)
        speech_duration = max(0.1, speech_end - speech_start)
        clip_ratio = _chinese_count(clip_text) / max(1, full_syllable_count)
        estimated_end = speech_start + speech_duration * clip_ratio
        end = _snap_to_nearby_silence(wav, sr, estimated_end)
        start = max(0.0, speech_start - 0.15)
        end = max(end, start + 1.0)

    start_i = int(start * sr)
    end_i = min(len(wav), int(end * sr))
    sf.write(str(output_path), wav[start_i:end_i], sr, subtype="PCM_16")
    return (end_i - start_i) / sr


def _speech_bounds(wav: np.ndarray, sr: int) -> tuple[float, float]:
    frame = max(1, int(sr * 0.02))
    hop = max(1, int(sr * 0.01))
    rms = _rms_frames(wav, frame, hop)
    threshold = max(0.006, float(np.percentile(rms, 70)) * 0.25)
    voiced = np.flatnonzero(rms > threshold)
    if len(voiced) == 0:
        return 0.0, len(wav) / sr
    return voiced[0] * hop / sr, min(len(wav) / sr, (voiced[-1] * hop + frame) / sr)


def _snap_to_nearby_silence(wav: np.ndarray, sr: int, estimated_end: float) -> float:
    frame = max(1, int(sr * 0.03))
    hop = max(1, int(sr * 0.01))
    rms = _rms_frames(wav, frame, hop)
    threshold = max(0.004, float(np.percentile(rms, 45)) * 0.8)
    start_t = max(0.0, estimated_end - 1.0)
    end_t = min(len(wav) / sr, estimated_end + 2.2)
    start_i = max(0, int(start_t * sr / hop))
    end_i = min(len(rms), int(end_t * sr / hop))
    candidates = np.flatnonzero(rms[start_i:end_i] < threshold)
    if len(candidates) == 0:
        return min(len(wav) / sr, estimated_end + 0.35)

    candidate_times = (candidates + start_i) * hop / sr
    # 优先选估算点之后的第一个停顿，避免把句子截短。
    after = candidate_times[candidate_times >= estimated_end]
    chosen = after[0] if len(after) else candidate_times[np.argmin(np.abs(candidate_times - estimated_end))]
    return min(len(wav) / sr, float(chosen + 0.25))


def _rms_frames(wav: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(wav) < frame:
        return np.array([float(np.sqrt(np.mean(np.square(wav))))])
    values = []
    for start in range(0, len(wav) - frame + 1, hop):
        chunk = wav[start:start + frame]
        values.append(float(np.sqrt(np.mean(np.square(chunk)))))
    return np.asarray(values, dtype=np.float32)


def _chinese_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


if __name__ == "__main__":
    main()
