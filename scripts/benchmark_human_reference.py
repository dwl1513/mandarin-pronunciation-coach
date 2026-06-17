"""用两套真人普通话范读交叉测试评分算法。

默认用 BBMU 真人范读作为参考音，把北京普通话学会同篇真人范读作为
待评测音频。这个实验用来校准：标准真人朗读应当得到较高分，如果分数
偏低，就说明声调、韵律或声学相似度规则过严。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR  # noqa: E402
from src.pipeline import assess  # noqa: E402


MANIFEST_PATH = DATA_DIR / "psc_human_dataset" / "manifest.json"
OUT_DIR = DATA_DIR / "psc_human_dataset" / "benchmarks"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行：uv run python scripts/collect_psc_reference_audio.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def audio_path(entry: dict, source_id: str) -> Path | None:
    for audio in entry.get("audios", []):
        if audio.get("source_id") == source_id:
            return ROOT / audio["wav_path"]
    return None


def score_pair(entry: dict,
               *,
               reference_source: str,
               user_source: str,
               use_asr: bool,
               asr_engine: str | None,
               f0_method: str) -> dict:
    ref_path = audio_path(entry, reference_source)
    user_path = audio_path(entry, user_source)
    if ref_path is None or user_path is None:
        raise FileNotFoundError(f"作品 {entry['number']:02d} 缺少配对音频")

    art = assess(
        user_path,
        entry["text"],
        use_asr=use_asr,
        use_tts_reference=True,
        reference_audio_path=ref_path,
        asr_engine=asr_engine,
        asr_fallback_to_local=False,
        prefer_model_alignment=False,
        f0_method=f0_method,
    )
    dims = art.report.get("dims", {})
    completeness = dims.get("completeness")
    acoustic_dims = [
        dims.get("accuracy", 0.0),
        dims.get("tone", 0.0),
        dims.get("fluency", 0.0),
        dims.get("prosody", 0.0),
    ]
    return {
        "number": entry["number"],
        "title": entry["title"],
        "syllable_count": entry.get("syllable_count", 0),
        "reference_source": reference_source,
        "user_source": user_source,
        "overall": art.report.get("overall", 0.0),
        "acoustic_overall": round(mean(acoustic_dims), 2),
        "accuracy": dims.get("accuracy", 0.0),
        "tone": dims.get("tone", 0.0),
        "fluency": dims.get("fluency", 0.0),
        "prosody": dims.get("prosody", 0.0),
        "completeness": completeness,
        "confidence": art.report.get("confidence", {}).get("overall", 0.0),
        "recognized": art.recognized_text,
        "reference_audio": str(ref_path.relative_to(ROOT)),
        "user_audio": str(user_path.relative_to(ROOT)),
    }


def _score_entry_for_pool(payload: tuple[dict, dict]) -> dict:
    """进程池入口函数。payload 只放可序列化对象，方便 macOS spawn。"""
    entry, options = payload
    return score_pair(
        entry,
        reference_source=options["reference_source"],
        user_source=options["user_source"],
        use_asr=options["use_asr"],
        asr_engine=options["asr_engine"],
        f0_method=options["f0_method"],
    )


def write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "human_reference_benchmark.csv"
    md_path = output_dir / "human_reference_benchmark.md"
    json_path = output_dir / "human_reference_benchmark.json"
    fields = [
        "number", "title", "syllable_count", "reference_source", "user_source",
        "overall", "acoustic_overall", "accuracy", "tone", "fluency",
        "prosody", "completeness", "confidence", "recognized",
        "reference_audio", "user_audio",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 真人标准普通话交叉评测",
        "",
        "参考音为一套真人范读，待测音为另一套同篇真人范读。",
        "",
        "| 编号 | 作品 | 字数 | 总分 | 声学均分 | 声韵母 | 声调 | 流利度 | 韵律 | 完整度 | 可信度 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        completeness = (
            ""
            if row["completeness"] is None
            else f"{row['completeness']:.1f}"
        )
        lines.append(
            f"| {row['number']} | {row['title']} | {row['syllable_count']} | "
            f"{row['overall']:.1f} | {row['acoustic_overall']:.1f} | "
            f"{row['accuracy']:.1f} | {row['tone']:.1f} | "
            f"{row['fluency']:.1f} | {row['prosody']:.1f} | "
            f"{completeness} | {row['confidence']:.1f} |"
        )
    if rows:
        lines.extend([
            "",
            "## 汇总",
            "",
            f"- 样本数：{len(rows)}",
            f"- 总分均值：{mean(row['overall'] for row in rows):.2f}",
            f"- 声学均分均值：{mean(row['acoustic_overall'] for row in rows):.2f}",
            f"- 声韵母均值：{mean(row['accuracy'] for row in rows):.2f}",
            f"- 声调均值：{mean(row['tone'] for row in rows):.2f}",
            f"- 流利度均值：{mean(row['fluency'] for row in rows):.2f}",
            f"- 韵律均值：{mean(row['prosody'] for row in rows):.2f}",
        ])
    lines.extend([
        "",
        f"CSV：`{csv_path.relative_to(ROOT)}`",
        f"JSON：`{json_path.relative_to(ROOT)}`",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[human-benchmark] CSV 已写入：{csv_path}")
    print(f"[human-benchmark] Markdown 已写入：{md_path}")
    print(f"[human-benchmark] JSON 已写入：{json_path}")


def run(args: argparse.Namespace) -> list[dict]:
    manifest = load_manifest()
    entries = manifest.get("entries", [])
    if args.numbers:
        wanted = {int(item) for item in args.numbers.split(",") if item.strip()}
        entries = [entry for entry in entries if int(entry["number"]) in wanted]
    if args.limit:
        entries = entries[:args.limit]

    rows: list[dict] = []
    if args.workers <= 1:
        for entry in entries:
            print(
                f"[human-benchmark] 作品 {entry['number']:02d}《{entry['title']}》 "
                f"{args.reference_source} -> {args.user_source}"
            )
            row = score_pair(
                entry,
                reference_source=args.reference_source,
                user_source=args.user_source,
                use_asr=args.use_asr,
                asr_engine=args.asr_engine,
                f0_method=args.f0_method,
            )
            print(
                f"  总分 {row['overall']:.1f}，声学均分 {row['acoustic_overall']:.1f}，"
                f"声调 {row['tone']:.1f}，韵律 {row['prosody']:.1f}"
            )
            rows.append(row)
    else:
        options = {
            "reference_source": args.reference_source,
            "user_source": args.user_source,
            "use_asr": args.use_asr,
            "asr_engine": args.asr_engine,
            "f0_method": args.f0_method,
        }
        print(
            f"[human-benchmark] 并发评测 {len(entries)} 篇，workers={args.workers}，"
            f"{args.reference_source} -> {args.user_source}"
        )
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_score_entry_for_pool, (entry, options)): entry
                for entry in entries
            }
            for future in as_completed(futures):
                entry = futures[future]
                row = future.result()
                rows.append(row)
                print(
                    f"[human-benchmark] 完成 {entry['number']:02d}《{entry['title']}》 "
                    f"总分 {row['overall']:.1f}，声调 {row['tone']:.1f}"
                )

    rows.sort(key=lambda item: int(item["number"]))

    write_outputs(rows, OUT_DIR)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="真人标准普通话交叉评测")
    parser.add_argument("--limit", type=int, default=6,
                        help="评测前 N 篇，默认 6 篇")
    parser.add_argument("--numbers", default="",
                        help="指定作品编号，例如 1,2,6,44")
    parser.add_argument("--reference-source", default="bbmu",
                        help="参考音来源，默认 bbmu")
    parser.add_argument("--user-source", default="beijing_putonghua",
                        help="待测音来源，默认 beijing_putonghua")
    parser.add_argument("--use-asr", action="store_true",
                        help="调用 ASR 计算完整度。默认关闭，避免长文评测被云端 ASR 耗时影响")
    parser.add_argument("--asr-engine", default="aliyun-asr",
                        help="ASR 引擎，默认 aliyun-asr")
    parser.add_argument("--f0-method", default="yin", choices=["yin", "pyin"],
                        help="F0 提取方法。yin 更快，pyin 更准但长文很慢")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发进程数，默认 1。建议 4；调用 ASR 时谨慎调大")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
