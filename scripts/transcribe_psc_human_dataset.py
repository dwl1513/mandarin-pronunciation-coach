"""用 ASR 为普通话水平测试真人范读数据集生成转写。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR  # noqa: E402
from src.asr.recognizer import recognize  # noqa: E402
from src.audio.capture import load_audio  # noqa: E402
from src.scoring.completeness import score_completeness  # noqa: E402


DATASET_DIR = DATA_DIR / "psc_human_dataset"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
TRANSCRIPT_DIR = DATASET_DIR / "transcripts"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {MANIFEST_PATH}，请先运行："
            "uv run python scripts/collect_psc_reference_audio.py"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def transcript_paths(source_id: str, number: int) -> tuple[Path, Path]:
    base = TRANSCRIPT_DIR / source_id
    return base / f"{number:03d}.json", base / f"{number:03d}.txt"


def collect_transcript_summary(manifest: dict) -> list[dict]:
    """从已有转写文件重新生成完整汇总，避免定点重试覆盖全量 summary。"""
    rows: list[dict] = []
    for entry in manifest.get("entries", []):
        number = int(entry["number"])
        for audio in entry.get("audios", []):
            json_path, txt_path = transcript_paths(audio["source_id"], number)
            if json_path.exists():
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if not txt_path.exists():
                    txt_path.write_text(
                        data.get("recognized_text", "") + "\n",
                        encoding="utf-8",
                    )
                rows.append(data)
    return rows


def refresh_transcript_score(data: dict, reference_text: str) -> dict:
    """用当前完整度规则刷新已有 ASR 转写的分数。"""
    if data.get("error"):
        return data
    completeness = score_completeness(data.get("recognized_text", ""), reference_text)
    data["char_coverage"] = completeness.coverage
    data["overall"] = completeness.overall
    return data


def transcribe_audio(audio_path: Path,
                     reference_text: str,
                     *,
                     engine: str) -> dict:
    wav = load_audio(audio_path)
    text = recognize(wav, engine=engine, fallback_to_local=False)
    completeness = score_completeness(text, reference_text)
    return {
        "recognized_text": text,
        "char_coverage": completeness.coverage,
        "overall": completeness.overall,
    }


def run(args: argparse.Namespace) -> list[dict]:
    manifest = load_manifest()
    entries = manifest.get("entries", [])
    if args.numbers:
        wanted = {int(item) for item in args.numbers.split(",") if item.strip()}
        entries = [entry for entry in entries if int(entry["number"]) in wanted]
    if args.limit:
        entries = entries[:args.limit]

    rows: list[dict] = []
    for entry in entries:
        number = int(entry["number"])
        title = entry["title"]
        for audio in entry.get("audios", []):
            source_id = audio["source_id"]
            if args.source and source_id != args.source:
                continue

            json_path, txt_path = transcript_paths(source_id, number)
            if json_path.exists() and not args.force:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                has_error = bool(data.get("error"))
                if not (args.retry_errors and has_error):
                    if args.refresh_scores:
                        data = refresh_transcript_score(data, entry["text"])
                        json_path.write_text(
                            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    if not txt_path.exists():
                        txt_path.write_text(
                            data.get("recognized_text", "") + "\n",
                            encoding="utf-8",
                        )
                    rows.append(data)
                    print(f"[asr] 跳过已有：{source_id} {number:03d}《{title}》")
                    continue
                print(f"[asr] 重试失败项：{source_id} {number:03d}《{title}》")

            audio_path = ROOT / audio["wav_path"]
            print(f"[asr] 识别：{source_id} {number:03d}《{title}》")
            try:
                result = transcribe_audio(
                    audio_path,
                    entry["text"],
                    engine=args.engine,
                )
                data = {
                    "number": number,
                    "title": title,
                    "source_id": source_id,
                    "source_name": audio.get("source_name", ""),
                    "audio_path": audio["wav_path"],
                    "engine": args.engine,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    **result,
                }
            except Exception as exc:
                data = {
                    "number": number,
                    "title": title,
                    "source_id": source_id,
                    "source_name": audio.get("source_name", ""),
                    "audio_path": audio["wav_path"],
                    "engine": args.engine,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "error": repr(exc),
                    "recognized_text": "",
                    "char_coverage": 0.0,
                    "overall": 0.0,
                }
                if args.stop_on_error:
                    raise

            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            txt_path.write_text(data["recognized_text"] + "\n", encoding="utf-8")
            rows.append(data)

    summary_rows = collect_transcript_summary(manifest)
    summary_path = TRANSCRIPT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[asr] 转写汇总已写入：{summary_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="普通话真人范读数据集 ASR 转写")
    parser.add_argument("--engine", default="aliyun-asr",
                        help="ASR 引擎，默认 aliyun-asr")
    parser.add_argument("--source", default="",
                        help="只识别某个来源，例如 bbmu 或 beijing_putonghua")
    parser.add_argument("--numbers", default="",
                        help="指定作品编号，例如 1,2,18")
    parser.add_argument("--limit", type=int, default=None,
                        help="只识别前 N 篇")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已有转写")
    parser.add_argument("--retry-errors", action="store_true",
                        help="只重试已有转写里带 error 的条目")
    parser.add_argument("--refresh-scores", action="store_true",
                        help="用当前完整度规则刷新已有转写分数，不重新调用 ASR")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="遇到 ASR 失败立即停止")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
