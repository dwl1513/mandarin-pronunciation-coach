"""抓取普通话水平测试真人范读音频并整理成正式数据集。"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR  # noqa: E402


BBMU_PAGE = "https://gh.bbmu.edu.cn/info/1147/3437.htm"
BEIJING_INDEX = "https://www.beijingputonghua.com/psc/ldzp/ldzp.htm"
BEIJING_BASE = "https://www.beijingputonghua.com/psc/ldzp/"

OUT_DIR = DATA_DIR / "psc_human_dataset"
MANIFEST_PATH = OUT_DIR / "manifest.json"


@dataclass
class SourceAudio:
    source_id: str
    source_name: str
    page_url: str
    mp3_url: str
    mp3_path: str
    wav_path: str


@dataclass
class PscEntry:
    number: int
    title: str
    text: str
    syllable_count: int
    audios: list[SourceAudio]


class TextExtractor(HTMLParser):
    """把网页片段转成适合解析的纯文本。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in {"p", "div", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.parts.append(data)

    def text(self) -> str:
        return unescape("".join(self.parts))


def fetch_bytes(url: str, *, timeout: int = 30) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/125 Safari/537.36"
        )
    }
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    lines = []
    for raw in parser.text().splitlines():
        line = re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("//", "")
    text = re.sub(r"[ \t]+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def count_chinese_syllables(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def extract_body_text(segment_html: str) -> str:
    plain = html_to_text(segment_html)
    lines = [line.strip() for line in plain.splitlines() if line.strip()]

    start = 0
    for idx, line in enumerate(lines):
        if "朗读基调" in line:
            start = idx + 1
            break
    else:
        for idx, line in enumerate(lines):
            if "【朗读提示】" in line:
                start = idx + 1
                break

    body: list[str] = []
    for line in lines[start:]:
        if "节选自" in line or line.startswith("选自"):
            break
        if "朗读提示" in line or "朗读基调" in line:
            continue
        body.append(line)

    return normalize_text("\n".join(body))


def parse_bbmu_entries() -> list[dict]:
    html = fetch_bytes(BBMU_PAGE).decode("utf-8", errors="replace")
    pattern = re.compile(
        r"作品\s*</span>\s*<span[^>]*>\s*"
        r"(?P<number>\d+)号\s*:?\s*《(?P<title>.*?)》\s*"
        r"<script>\s*showVsbAudio\('(?P<audio>[^']+?\.mp3)'",
        re.S,
    )
    matches = list(pattern.finditer(html))
    if len(matches) != 50:
        raise RuntimeError(f"BBMU 页面解析到 {len(matches)} 条音频，预期 50 条")

    entries: list[dict] = []
    for idx, match in enumerate(matches):
        segment_start = match.end()
        segment_end = matches[idx + 1].start() if idx + 1 < len(matches) else html.find("朗读示范人简介", segment_start)
        if segment_end < 0:
            segment_end = len(html)
        title = re.sub(r"\s+", "", match.group("title"))
        number = int(match.group("number"))
        text = extract_body_text(html[segment_start:segment_end])
        entries.append({
            "number": number,
            "title": title,
            "text": text,
            "bbmu_mp3_url": urljoin(BBMU_PAGE, match.group("audio")),
        })
    return entries


def parse_beijing_audio_urls(numbers: Iterable[int]) -> dict[int, str]:
    urls: dict[int, str] = {}
    for number in numbers:
        page_url = urljoin(BEIJING_BASE, f"zp{number:02d}.htm")
        try:
            html = fetch_bytes(page_url).decode("big5", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[collect] 北京普通话学会第 {number:02d} 篇页面读取失败：{exc}")
            continue
        match = re.search(r'<source\s+src="([^"]+?\.mp3)"', html, re.I)
        if match:
            urls[number] = urljoin(page_url, match.group(1))
    return urls


def download_file(url: str, path: Path, *, force: bool = False) -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if shutil.which("curl"):
        cmd = [
            "curl",
            "-L",
            "--fail",
            "-sS",
            "--connect-timeout",
            "10",
            "--max-time",
            "45",
            "--retry",
            "2",
            "--retry-delay",
            "1",
            "-A",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/125",
            "-o",
            str(tmp),
            url,
        ]
        subprocess.run(cmd, check=True)
    else:
        data = fetch_bytes(url, timeout=35)
        tmp.write_bytes(data)
    tmp.replace(path)


def convert_to_wav(mp3_path: Path, wav_path: Path, *, force: bool = False) -> None:
    if wav_path.exists() and wav_path.stat().st_size > 0 and not force:
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def build_manifest(limit: int | None, force: bool) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_entries = parse_bbmu_entries()
    if limit:
        raw_entries = raw_entries[:limit]

    beijing_urls = parse_beijing_audio_urls(entry["number"] for entry in raw_entries)
    entries: list[PscEntry] = []

    for raw in raw_entries:
        number = raw["number"]
        title = raw["title"]
        print(f"[collect] 作品 {number:02d}《{title}》")
        audios: list[SourceAudio] = []

        bbmu_mp3 = OUT_DIR / "audio" / "bbmu" / f"{number:03d}.mp3"
        bbmu_wav = OUT_DIR / "audio" / "bbmu" / f"{number:03d}.wav"
        download_file(raw["bbmu_mp3_url"], bbmu_mp3, force=force)
        convert_to_wav(bbmu_mp3, bbmu_wav, force=force)
        audios.append(SourceAudio(
            source_id="bbmu",
            source_name="蚌埠医科大学工会",
            page_url=BBMU_PAGE,
            mp3_url=raw["bbmu_mp3_url"],
            mp3_path=str(bbmu_mp3.relative_to(ROOT)),
            wav_path=str(bbmu_wav.relative_to(ROOT)),
        ))

        beijing_url = beijing_urls.get(number)
        if beijing_url:
            bj_mp3 = OUT_DIR / "audio" / "beijing_putonghua" / f"{number:03d}.mp3"
            bj_wav = OUT_DIR / "audio" / "beijing_putonghua" / f"{number:03d}.wav"
            download_file(beijing_url, bj_mp3, force=force)
            convert_to_wav(bj_mp3, bj_wav, force=force)
            audios.append(SourceAudio(
                source_id="beijing_putonghua",
                source_name="北京普通话学会",
                page_url=urljoin(BEIJING_BASE, f"zp{number:02d}.htm"),
                mp3_url=beijing_url,
                mp3_path=str(bj_mp3.relative_to(ROOT)),
                wav_path=str(bj_wav.relative_to(ROOT)),
            ))

        entries.append(PscEntry(
            number=number,
            title=title,
            text=raw["text"],
            syllable_count=count_chinese_syllables(raw["text"]),
            audios=audios,
        ))

        # 对小站点轻一点，避免短时间请求过密。
        time.sleep(0.08)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": "普通话水平测试新版 50 篇真人范读本地实验清单",
        "dataset_dir": str(OUT_DIR.relative_to(ROOT)),
        "sources": [
            {
                "source_id": "bbmu",
                "name": "蚌埠医科大学工会",
                "url": BBMU_PAGE,
                "note": "页面标注含朗读示范，示范人为国家级普通话水平测试员、普通话一级甲等。",
            },
            {
                "source_id": "beijing_putonghua",
                "name": "北京普通话学会",
                "url": BEIJING_INDEX,
                "note": "页面标注为普通话水平测试 2021 年版朗读作品 50 篇连声音档。",
            },
        ],
        "entries": [
            {
                **{k: v for k, v in asdict(entry).items() if k != "audios"},
                "audios": [asdict(audio) for audio in entry.audios],
            }
            for entry in entries
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取普通话水平测试真人范读音频")
    parser.add_argument("--limit", type=int, default=None,
                        help="只抓取前 N 篇，调试时使用")
    parser.add_argument("--force", action="store_true",
                        help="重新下载并重新转码已有音频")
    args = parser.parse_args()

    manifest = build_manifest(args.limit, args.force)
    print(f"[collect] 清单已写入：{MANIFEST_PATH}")
    print(f"[collect] 共整理 {len(manifest['entries'])} 篇作品")


if __name__ == "__main__":
    main()
