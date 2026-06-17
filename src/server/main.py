"""普通话发音教练 FastAPI 后端。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import ALIYUN_ASR_MODEL, CACHE_DIR, MIMO_TTS_MODEL
from src.pipeline import assess
from src.server.clipper import router as clipper_router
from src.server.examples import example_audio_path, load_examples
from src.server.render import save_assessment_assets
from src.server.schemas import (
    AssessmentJobCreated,
    AssessmentJobStatus,
    AssessmentResponse,
    ExampleItem,
)


RESULT_ROOT = CACHE_DIR / "web_results"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
JOBS: dict[str, dict[str, Any]] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="Mandarin Pronunciation Coach API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(clipper_router)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "fastapi",
        "asr_model": ALIYUN_ASR_MODEL,
        "tts_model": MIMO_TTS_MODEL,
        "result_dir": str(RESULT_ROOT),
    }


@app.get("/api/examples", response_model=list[ExampleItem])
def examples() -> list[dict[str, Any]]:
    return load_examples()


@app.get("/api/examples/{example_id}/audio")
def get_example_audio(example_id: str) -> FileResponse:
    path = example_audio_path(example_id)
    if path is None:
        raise HTTPException(status_code=404, detail="例句标准音不存在")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/assess", response_model=AssessmentJobCreated)
async def assess_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    reference_text: str = Form(...),
    example_id: str | None = Form(default=None),
    asr_engine: str = Form(default="aliyun-asr"),
    tts_engine: str = Form(default="mimo-tts,aliyun-tts"),
    tts_voice: str | None = Form(default=None),
) -> AssessmentJobCreated:
    text = reference_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="参考文本不能为空")

    assessment_id = uuid.uuid4().hex
    result_dir = RESULT_ROOT / assessment_id
    result_dir.mkdir(parents=True, exist_ok=True)

    suffix = _upload_suffix(audio.filename, audio.content_type)
    upload_path = result_dir / f"upload{suffix}"
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="录音文件为空")
    upload_path.write_bytes(content)

    _set_job(
        assessment_id,
        status="queued",
        stage="upload",
        message="录音已上传，等待评测任务启动",
        progress=5,
    )
    background_tasks.add_task(
        _submit_assessment_job,
        assessment_id,
        upload_path,
        text,
        example_id,
        asr_engine,
        tts_engine,
        tts_voice,
    )
    return AssessmentJobCreated(
        id=assessment_id,
        status_url=f"/api/assess-jobs/{assessment_id}",
    )


@app.get("/api/assess-jobs/{assessment_id}", response_model=AssessmentJobStatus)
def get_assessment_job(assessment_id: str) -> dict[str, Any]:
    job = JOBS.get(assessment_id)
    if job is None:
        raise HTTPException(status_code=404, detail="评测任务不存在")
    return job


def _submit_assessment_job(
    assessment_id: str,
    upload_path: Path,
    text: str,
    example_id: str | None,
    asr_engine: str,
    tts_engine: str,
    tts_voice: str | None,
) -> None:
    EXECUTOR.submit(
        _run_assessment_job,
        assessment_id,
        upload_path,
        text,
        example_id,
        asr_engine,
        tts_engine,
        tts_voice,
    )


def _run_assessment_job(
    assessment_id: str,
    upload_path: Path,
    text: str,
    example_id: str | None,
    asr_engine: str,
    tts_engine: str,
    tts_voice: str | None,
) -> None:
    reference_audio_path = example_audio_path(example_id) if example_id else None
    try:
        _set_job(
            assessment_id,
            status="running",
            stage="asr_tts",
            message="正在调用 ASR、标准音和声学评分链路",
            progress=20,
        )
        print(
            "[server] 开始评测 "
            f"id={assessment_id} example={example_id or '-'} "
            f"ref_audio={'yes' if reference_audio_path else 'no'} "
            f"asr={asr_engine or '-'} tts={tts_engine or '-'}",
            flush=True,
        )
        artifacts = assess(
            upload_path,
            text,
            use_asr=True,
            use_tts_reference=True,
            asr_engine=asr_engine or None,
            tts_engine=tts_engine or None,
            tts_voice=tts_voice or None,
            reference_audio_path=reference_audio_path,
            asr_fallback_to_local=False,
            prefer_model_alignment=False,
        )
        _set_job(
            assessment_id,
            status="running",
            stage="render",
            message="评分完成，正在生成波形、频谱和 F0 图",
            progress=80,
        )
        print(f"[server] 评测完成 id={assessment_id}，开始生成声学证据", flush=True)
        result_dir = RESULT_ROOT / assessment_id
        asset_names = save_assessment_assets(artifacts, result_dir)
        print(f"[server] 声学证据生成完成 id={assessment_id}", flush=True)
    except Exception as exc:
        _set_job(
            assessment_id,
            status="failed",
            stage="failed",
            message="评测失败",
            progress=100,
            error=str(exc),
        )
        print(f"[server] 评测失败 id={assessment_id}: {exc}", flush=True)
        return

    response = _build_assessment_response(
        assessment_id=assessment_id,
        text=text,
        example_id=example_id,
        artifacts=artifacts,
        asset_names=asset_names,
    )
    _set_job(
        assessment_id,
        status="done",
        stage="done",
        message="评测完成",
        progress=100,
        result=response,
    )


def _build_assessment_response(
    assessment_id: str,
    text: str,
    example_id: str | None,
    artifacts: Any,
    asset_names: dict[str, str | None],
) -> AssessmentResponse:
    report = _json_ready(artifacts.report)
    evidence = {
        "user_audio_url": _result_url(assessment_id, asset_names["user_audio"]),
        "reference_audio_url": (
            None
            if asset_names["reference_audio"] is None
            else _result_url(assessment_id, asset_names["reference_audio"])
        ),
        "waveform_url": _result_url(assessment_id, asset_names["waveform"]),
        "spectrogram_url": _result_url(assessment_id, asset_names["spectrogram"]),
        "f0_url": _result_url(assessment_id, asset_names["f0"]),
    }

    return AssessmentResponse(
        id=assessment_id,
        reference_text=text,
        recognized_text=artifacts.recognized_text,
        overall=float(report.get("overall", 0.0)),
        dims=report.get("dims", {}),
        confidence=report.get("confidence", {}),
        fluency_detail=report.get("fluency_detail", {}),
        notes=report.get("notes", []),
        per_syllable=report.get("per_syllable", []),
        advice=_build_advice(report, artifacts.recognized_text, example_id),
        evidence=evidence,
        markdown=str(report.get("markdown", "")),
    )


def _set_job(assessment_id: str, **updates: Any) -> None:
    current = JOBS.get(assessment_id, {
        "id": assessment_id,
        "status": "queued",
        "stage": "upload",
        "message": "等待处理",
        "progress": 0,
        "error": None,
        "result": None,
    })
    current.update(updates)
    JOBS[assessment_id] = current


@app.get("/api/results/{assessment_id}/{file_name}")
def get_result_file(assessment_id: str, file_name: str) -> FileResponse:
    if "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=404, detail="文件不存在")

    path = (RESULT_ROOT / assessment_id / file_name).resolve()
    root = RESULT_ROOT.resolve()
    if root not in path.parents:
        raise HTTPException(status_code=404, detail="文件不存在")
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    media_type = "application/octet-stream"
    if path.suffix == ".wav":
        media_type = "audio/wav"
    elif path.suffix == ".png":
        media_type = "image/png"
    return FileResponse(path, media_type=media_type)


def _upload_suffix(filename: str | None, content_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix:
        return suffix
    if content_type == "audio/wav":
        return ".wav"
    if content_type == "audio/mp4":
        return ".mp4"
    if content_type == "audio/mpeg":
        return ".mp3"
    return ".webm"


def _result_url(assessment_id: str, file_name: str | None) -> str:
    if file_name is None:
        return ""
    return f"/api/results/{assessment_id}/{file_name}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _build_advice(report: dict[str, Any],
                  recognized_text: str,
                  example_id: str | None) -> list[dict[str, str]]:
    dims = report.get("dims", {})
    per_syllable = report.get("per_syllable", [])
    notes = report.get("notes", [])
    overall = float(report.get("overall", 0.0))
    advice: list[dict[str, str]] = []

    weak_dims = sorted(
        ((name, float(score)) for name, score in dims.items()),
        key=lambda item: item[1],
    )
    dim_names = {
        "accuracy": "声韵母",
        "tone": "声调",
        "fluency": "流利度",
        "prosody": "韵律",
        "completeness": "完整度",
    }

    for name, score in weak_dims[:2]:
        if score >= 88:
            continue
        title = f"{dim_names.get(name, name)}需要加强"
        detail = _dimension_advice(name, score)
        advice.append({"level": "warn" if score >= 70 else "bad",
                       "title": title, "detail": detail})

    weak_chars = [
        item for item in per_syllable
        if item.get("note") or float(item.get("acc_score", 100)) < 85
        or float(item.get("tone_score", 100)) < 80
        or not item.get("completeness_ok", True)
    ]
    if weak_chars:
        chars = "、".join(str(item.get("char", "")) for item in weak_chars[:5])
        notes_show = "；".join(
            str(item.get("note") or "分数偏低") for item in weak_chars[:3]
        )
        advice.append({
            "level": "warn",
            "title": f"重点复练：{chars}",
            "detail": notes_show,
        })

    if notes:
        advice.append({
            "level": "warn",
            "title": "录音质量提示",
            "detail": "；".join(str(item) for item in notes[:2]),
        })

    if not recognized_text:
        advice.append({
            "level": "bad",
            "title": "ASR 没有返回识别文本",
            "detail": "请确认录音音量、网络和 ASR Key。完整度分数会受到影响。",
        })

    if not advice:
        advice.append({
            "level": "good",
            "title": "整体发音稳定",
            "detail": (
                "本次朗读的声韵母、声调、流利度和完整度都比较好，可以继续换更长的句子。"
                if overall >= 90 else
                "本次没有明显单字错误，建议继续关注语速和自然停顿。"
            ),
        })

    if example_id and example_id != "custom":
        advice.append({
            "level": "good",
            "title": "练习材料已记录",
            "detail": "本次结果使用固定例句评测，适合答辩现场复现。"
        })

    return advice[:4]


def _dimension_advice(name: str, score: float) -> str:
    if name == "tone":
        return f"声调与标准音相似度为 {score:.1f}，建议对照 F0 曲线看高低、升降和三声下探。"
    if name == "accuracy":
        return f"声韵母分为 {score:.1f}，建议优先看逐字表里的声母、韵母和发声覆盖。"
    if name == "fluency":
        return f"流利度分为 {score:.1f}，建议减少长停顿，保持中等语速。"
    if name == "prosody":
        return f"韵律分为 {score:.1f}，建议听标准音，模仿句中重音和自然停连。"
    if name == "completeness":
        return f"完整度分为 {score:.1f}，可能存在漏读、吞字或 ASR 识别失败。"
    return f"当前分数为 {score:.1f}，建议结合声学证据复查。"
