"""批量生成前端例句库的标准发音。

默认读取 `frontend/src/data/practice-examples.json`，并把合成出的 WAV
复制到 `frontend/public/audio/examples/`。前端通过 `/audio/examples/*.wav`
直接播放这些静态文件。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feedback.tts import synth_reference  # noqa: E402


EXAMPLES_JSON = PROJECT_ROOT / "frontend/src/data/practice-examples.json"
OUTPUT_DIR = PROJECT_ROOT / "frontend/public/audio/examples"


def _load_examples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"例句库格式错误：{path}")
    return data


def generate_examples(engine: str, voice: str | None, force: bool) -> None:
    examples = _load_examples(EXAMPLES_JSON)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, example in enumerate(examples, start=1):
        example_id = str(example["id"])
        text = str(example["text"])
        output_path = OUTPUT_DIR / f"{example_id}.wav"

        if output_path.exists() and not force:
            print(f"[{index}/{len(examples)}] 已存在：{output_path.relative_to(PROJECT_ROOT)}")
            continue

        print(f"[{index}/{len(examples)}] 生成：{example_id} | {text}")
        source_path = synth_reference(text, engine=engine, voice=voice)
        shutil.copyfile(source_path, output_path)
        print(f"  写入：{output_path.relative_to(PROJECT_ROOT)}")

    print("例句库标准发音生成完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成前端例句库标准发音")
    parser.add_argument(
        "--engine",
        default="mimo-tts",
        choices=["mimo-tts", "aliyun-tts", "edge-tts"],
        help="TTS 引擎，默认 mimo-tts",
    )
    parser.add_argument("--voice", default=None, help="TTS 音色，不传则读取 .env 默认值")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的音频文件")
    args = parser.parse_args()

    generate_examples(engine=args.engine, voice=args.voice, force=args.force)


if __name__ == "__main__":
    main()
