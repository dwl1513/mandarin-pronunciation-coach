"""FastAPI 响应模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExampleItem(BaseModel):
    id: str
    title: str
    level: str
    focus: str
    scene: str
    duration: str
    text: str
    tags: list[str] = Field(default_factory=list)
    audioPath: str = ""


class AdviceItem(BaseModel):
    level: Literal["good", "warn", "bad"]
    title: str
    detail: str


class EvidenceLinks(BaseModel):
    user_audio_url: str
    reference_audio_url: str | None = None
    waveform_url: str
    spectrogram_url: str
    f0_url: str


class AssessmentResponse(BaseModel):
    id: str
    reference_text: str
    recognized_text: str
    overall: float
    dims: dict[str, float]
    confidence: dict[str, Any] = Field(default_factory=dict)
    fluency_detail: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    per_syllable: list[dict[str, Any]] = Field(default_factory=list)
    advice: list[AdviceItem] = Field(default_factory=list)
    evidence: EvidenceLinks
    markdown: str

