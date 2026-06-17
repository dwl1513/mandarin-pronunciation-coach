"""PSC 例句音频片段人工标注工具。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from config import PROJECT_ROOT


router = APIRouter()

DATA_DIR = PROJECT_ROOT / "data/psc_human_dataset"
MANIFEST_PATH = DATA_DIR / "manifest.json"
SPECS_PATH = DATA_DIR / "example_clip_specs.json"
CLIPS_PATH = DATA_DIR / "example_clips.json"


class ClipUpdate(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)


@router.get("/tools/clipper", response_class=HTMLResponse)
def clipper_page() -> str:
    return _HTML


@router.get("/api/tools/clipper/items")
def list_items() -> list[dict[str, Any]]:
    entries = _manifest_entries()
    clips = _load_clips()
    out = []
    for spec in _load_specs():
        entry = entries[int(spec["number"])]
        audio_path = _bbmu_audio_path(entry)
        duration = _audio_duration(audio_path)
        text = _take_sentences(str(entry["text"]), int(spec["sentences"]))
        clip = clips.get(str(spec["id"]), {})
        out.append({
            "id": spec["id"],
            "number": spec["number"],
            "title": spec["title"],
            "text": text,
            "duration": duration,
            "start": clip.get("start"),
            "end": clip.get("end"),
            "audio_url": f"/api/tools/clipper/source/{spec['id']}",
            "waveform_url": f"/api/tools/clipper/waveform/{spec['id']}",
        })
    return out


@router.get("/api/tools/clipper/source/{example_id}")
def source_audio(example_id: str) -> FileResponse:
    spec = _find_spec(example_id)
    entries = _manifest_entries()
    audio_path = _bbmu_audio_path(entries[int(spec["number"])])
    return FileResponse(audio_path, media_type="audio/wav")


@router.get("/api/tools/clipper/waveform/{example_id}")
def waveform(example_id: str) -> dict[str, Any]:
    spec = _find_spec(example_id)
    entries = _manifest_entries()
    audio_path = _bbmu_audio_path(entries[int(spec["number"])])
    wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    target_points = 2400
    block = max(1, len(wav) // target_points)
    usable = wav[:len(wav) // block * block]
    shaped = usable.reshape(-1, block)
    peaks = np.max(np.abs(shaped), axis=1)
    return {
        "duration": len(wav) / sr,
        "peaks": peaks.round(4).tolist(),
    }


@router.post("/api/tools/clipper/clips/{example_id}")
def save_clip(example_id: str, update: ClipUpdate) -> dict[str, Any]:
    if update.end <= update.start:
        raise HTTPException(status_code=400, detail="结束时间必须大于开始时间")
    spec = _find_spec(example_id)
    entries = _manifest_entries()
    duration = _audio_duration(_bbmu_audio_path(entries[int(spec["number"])]))
    if update.end > duration:
        raise HTTPException(status_code=400, detail="结束时间超过音频总时长")

    clips = _load_clips()
    clips[example_id] = {
        "id": example_id,
        "start": round(update.start, 3),
        "end": round(update.end, 3),
    }
    ordered = [clips[str(item["id"])] for item in _load_specs() if str(item["id"]) in clips]
    CLIPS_PATH.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return clips[example_id]


def _load_specs() -> list[dict[str, Any]]:
    return json.loads(SPECS_PATH.read_text(encoding="utf-8"))


def _manifest_entries() -> dict[int, dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {int(item["number"]): item for item in manifest["entries"]}


def _load_clips() -> dict[str, dict[str, Any]]:
    if not CLIPS_PATH.exists():
        return {}
    data = json.loads(CLIPS_PATH.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in data if isinstance(item, dict) and "id" in item}


def _find_spec(example_id: str) -> dict[str, Any]:
    for item in _load_specs():
        if item["id"] == example_id:
            return item
    raise HTTPException(status_code=404, detail="例句不存在")


def _bbmu_audio_path(entry: dict[str, Any]) -> Path:
    for audio in entry["audios"]:
        if audio["source_id"] == "bbmu":
            return PROJECT_ROOT / audio["wav_path"]
    raise HTTPException(status_code=404, detail="真人范读音频不存在")


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def _take_sentences(text: str, count: int) -> str:
    cleaned = re.sub(r"\s+", "", text).replace("〞", "”")
    parts = [part for part in re.split(r"(?<=[。！？])", cleaned) if part]
    return "".join(parts[:count]).strip()


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PSC 例句截取标注</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "PingFang SC", "Helvetica Neue", Arial, sans-serif; color: #111; background: #f6f6f4; }
    header { height: 58px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; border-bottom: 1px solid #ddd; background: #fff; position: sticky; top: 0; z-index: 5; }
    main { display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 16px; padding: 16px; }
    aside, section { background: #fff; border: 1px solid #ddd; border-radius: 8px; }
    aside { max-height: calc(100vh - 90px); overflow: auto; }
    button { font: inherit; cursor: pointer; }
    .item { width: 100%; text-align: left; border: 0; border-bottom: 1px solid #eee; background: #fff; padding: 12px; }
    .item.active { background: #111; color: #fff; }
    .item small { display: block; opacity: .65; margin-top: 4px; line-height: 1.4; }
    .panel { padding: 16px; }
    .title { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
    .title h1 { margin: 0; font-size: 22px; }
    .text { margin: 12px 0; padding: 14px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; font-size: 20px; line-height: 1.8; }
    canvas { display: block; width: 100%; height: 220px; border: 1px solid #ccc; border-radius: 8px; background: #fff; }
    .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
    input[type="number"] { width: 110px; height: 36px; border: 1px solid #ccc; border-radius: 6px; padding: 0 8px; }
    .btn { height: 36px; border: 1px solid #111; border-radius: 6px; padding: 0 12px; background: #111; color: #fff; }
    .btn.secondary { background: #fff; color: #111; }
    .hint { color: #666; font-size: 13px; line-height: 1.6; }
    audio { width: 100%; margin-top: 12px; }
    .saved { color: #0a7f37; }
    .warn { color: #b45309; }
  </style>
</head>
<body>
  <header>
    <strong>PSC 例句截取标注</strong>
    <span class="hint">拖动波形：左键设开始，右键设结束；也可以直接输入秒数。</span>
  </header>
  <main>
    <aside id="list"></aside>
    <section class="panel">
      <div class="title">
        <h1 id="title">加载中</h1>
        <span id="duration" class="hint"></span>
      </div>
      <div id="text" class="text"></div>
      <canvas id="wave" width="1400" height="260"></canvas>
      <audio id="audio" controls></audio>
      <div class="row">
        <label>开始 <input id="start" type="number" step="0.01" min="0" /></label>
        <label>结束 <input id="end" type="number" step="0.01" min="0" /></label>
        <button class="btn secondary" id="play-full">播放原音</button>
        <button class="btn secondary" id="play-clip">预听片段</button>
        <button class="btn" id="save">保存时间戳</button>
        <span id="status" class="hint"></span>
      </div>
      <p class="hint">
        使用建议：先点列表选材料，播放原音；在波形上左键点片段开头、右键点片段结尾；点“预听片段”检查；满意后保存。
        保存后回到终端运行 <code>uv run python scripts/generate_psc_example_assets.py</code> 重新导出前端标准音。
      </p>
    </section>
  </main>
  <script>
    const listEl = document.querySelector('#list')
    const titleEl = document.querySelector('#title')
    const textEl = document.querySelector('#text')
    const durationEl = document.querySelector('#duration')
    const canvas = document.querySelector('#wave')
    const ctx = canvas.getContext('2d')
    const audio = document.querySelector('#audio')
    const startInput = document.querySelector('#start')
    const endInput = document.querySelector('#end')
    const statusEl = document.querySelector('#status')
    let items = []
    let current = null
    let waveform = null

    async function boot() {
      items = await fetch('/api/tools/clipper/items').then(r => r.json())
      renderList()
      select(items[0])
    }

    function renderList() {
      listEl.innerHTML = ''
      for (const item of items) {
        const btn = document.createElement('button')
        btn.className = 'item' + (current?.id === item.id ? ' active' : '')
        btn.innerHTML = `<strong>${item.title}</strong><small>${item.id}<br>${item.start ?? '-'} → ${item.end ?? '-'}</small>`
        btn.onclick = () => select(item)
        listEl.appendChild(btn)
      }
    }

    async function select(item) {
      current = item
      titleEl.textContent = item.title
      textEl.textContent = item.text
      durationEl.textContent = `${item.duration.toFixed(2)} 秒`
      audio.src = item.audio_url
      startInput.value = item.start ?? 0
      endInput.value = item.end ?? Math.min(item.duration, 12).toFixed(2)
      waveform = await fetch(item.waveform_url).then(r => r.json())
      renderList()
      draw()
    }

    function draw() {
      if (!waveform) return
      const { peaks, duration } = waveform
      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)
      ctx.fillStyle = '#fff'
      ctx.fillRect(0, 0, w, h)
      ctx.strokeStyle = '#ddd'
      ctx.beginPath()
      ctx.moveTo(0, h / 2)
      ctx.lineTo(w, h / 2)
      ctx.stroke()
      ctx.strokeStyle = '#111'
      ctx.beginPath()
      peaks.forEach((p, i) => {
        const x = i / (peaks.length - 1) * w
        const y1 = h / 2 - p * h * 0.46
        const y2 = h / 2 + p * h * 0.46
        ctx.moveTo(x, y1)
        ctx.lineTo(x, y2)
      })
      ctx.stroke()

      const start = Number(startInput.value || 0)
      const end = Number(endInput.value || 0)
      const sx = start / duration * w
      const ex = end / duration * w
      ctx.fillStyle = 'rgba(0,0,0,.10)'
      ctx.fillRect(sx, 0, Math.max(1, ex - sx), h)
      ctx.fillStyle = '#0a7f37'
      ctx.fillRect(sx - 1, 0, 2, h)
      ctx.fillStyle = '#b91c1c'
      ctx.fillRect(ex - 1, 0, 2, h)
    }

    canvas.addEventListener('click', (event) => {
      const t = event.offsetX / canvas.clientWidth * waveform.duration
      startInput.value = t.toFixed(3)
      draw()
    })
    canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault()
      const t = event.offsetX / canvas.clientWidth * waveform.duration
      endInput.value = t.toFixed(3)
      draw()
    })
    startInput.addEventListener('input', draw)
    endInput.addEventListener('input', draw)

    document.querySelector('#play-full').onclick = () => {
      audio.currentTime = 0
      audio.play()
    }
    document.querySelector('#play-clip').onclick = () => {
      const start = Number(startInput.value || 0)
      const end = Number(endInput.value || 0)
      audio.currentTime = start
      audio.play()
      const timer = setInterval(() => {
        if (audio.currentTime >= end || audio.paused) {
          audio.pause()
          clearInterval(timer)
        }
      }, 40)
    }
    document.querySelector('#save').onclick = async () => {
      statusEl.textContent = '保存中'
      statusEl.className = 'hint warn'
      const resp = await fetch(`/api/tools/clipper/clips/${current.id}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ start: Number(startInput.value), end: Number(endInput.value) }),
      })
      const payload = await resp.json()
      if (!resp.ok) {
        statusEl.textContent = payload.detail || '保存失败'
        statusEl.className = 'hint warn'
        return
      }
      current.start = payload.start
      current.end = payload.end
      statusEl.textContent = '已保存'
      statusEl.className = 'hint saved'
      renderList()
    }

    boot()
  </script>
</body>
</html>
"""
