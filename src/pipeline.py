"""End-to-end assessment pipeline (M1 → M5).

This module is the **single source of truth** for "given user audio + reference
text, produce a score report".  The Gradio UI and the CLI both call this; the
tests pin its return shape so future refactors won't silently break the rest
of the project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from src.asr.aligner import align
from src.asr.recognizer import recognize
from src.audio.preprocess import PreprocessResult, preprocess
from src.feedback.report import build_report
from src.feedback.tts import synth_reference
from src.features.pitch import extract_f0
from src.features.spectral import extract_mfcc
from src.scoring.accuracy import score_accuracy
from src.scoring.aggregator import aggregate
from src.scoring.completeness import score_completeness
from src.scoring.confidence import score_confidence
from src.scoring.fluency import score_fluency
from src.scoring.prosody import score_prosody
from src.scoring.tone import score_tone


@dataclass
class ReferenceArtifacts:
    """One synthesized reference voice and its extracted features."""

    engine: str
    voice: Optional[str]
    audio_path: Path
    pre: PreprocessResult
    mfcc: np.ndarray
    alignment: list
    f0: np.ndarray
    f0_times: np.ndarray
    voiced: np.ndarray


@dataclass
class AssessmentArtifacts:
    """Everything the UI / report needs in one place."""

    report: Dict[str, Any]                       # build_report() output
    user_pre: PreprocessResult
    ref_pre: Optional[PreprocessResult] = None
    user_alignment: list = field(default_factory=list)
    ref_alignment: list = field(default_factory=list)
    recognized_text: str = ""
    reference_text: str = ""
    user_f0: Optional[np.ndarray] = None
    user_f0_times: Optional[np.ndarray] = None
    user_voiced: Optional[np.ndarray] = None
    ref_f0: Optional[np.ndarray] = None
    ref_f0_times: Optional[np.ndarray] = None
    ref_voiced: Optional[np.ndarray] = None
    ref_audio_path: Optional[Path] = None


def _parse_tts_engines(tts_engine: Optional[str]) -> list[tuple[str | None, str | None]]:
    """Parse comma-separated engines; optional voice can use engine:voice."""
    if not tts_engine:
        return [(None, None)]
    out: list[tuple[str | None, str | None]] = []
    for raw in tts_engine.split(","):
        item = raw.strip()
        if not item:
            continue
        engine, sep, voice = item.partition(":")
        out.append((engine.strip() or None, voice.strip() if sep and voice.strip() else None))
    return out or [(None, None)]


def _collect_reference(reference_text: str,
                       engine: Optional[str],
                       voice: Optional[str]) -> ReferenceArtifacts:
    ref_audio_path = synth_reference(reference_text, engine=engine, voice=voice)
    ref_pre = preprocess(ref_audio_path)
    ref_mfcc = extract_mfcc(ref_pre.wav)
    ref_f0, ref_times, ref_voiced = extract_f0(ref_pre.raw_wav)
    ref_alignment = align(ref_pre.raw_wav, reference_text,
                          vad_segments=ref_pre.vad_segments)
    return ReferenceArtifacts(
        engine=engine or "default",
        voice=voice,
        audio_path=ref_audio_path,
        pre=ref_pre,
        mfcc=ref_mfcc,
        alignment=ref_alignment,
        f0=ref_f0,
        f0_times=ref_times,
        voiced=ref_voiced,
    )


def _best_score(scores: list, prefer_per_syllable: bool = True):
    """Pick the highest-overall score object from several reference voices."""
    if not scores:
        return None
    if prefer_per_syllable and all(getattr(s, "per_syllable", None) for s in scores):
        # 各参考音长度一致时，逐字取最高分，更符合 multi-reference 思路。
        base = max(scores, key=lambda s: float(getattr(s, "overall", 0.0)))
        n = min(len(s.per_syllable) for s in scores)
        picked = []
        for i in range(n):
            picked.append(max((s.per_syllable[i] for s in scores),
                              key=lambda item: float(getattr(item, "score", 0.0))))
        base.per_syllable = picked
        base.overall = float(np.mean([float(getattr(item, "score", 0.0)) for item in picked]))
        return base
    return max(scores, key=lambda s: float(getattr(s, "overall", 0.0)))


def assess(user_audio: Union[str, Path, np.ndarray, tuple],
           reference_text: str,
           *,
           use_asr: bool = True,
           use_tts_reference: bool = True,
           asr_engine: Optional[str] = None,
           tts_engine: Optional[str] = None,
           tts_voice: Optional[str] = None) -> AssessmentArtifacts:
    """Score a single utterance against a reference text.

    Args:
        user_audio:         path / ndarray / (sr, ndarray) — anything
                            `load_audio` understands.
        reference_text:     what the user was supposed to read.
        use_asr:            run wav2vec2 ASR for completeness scoring.
                            Disable to skip the (slow) model load when only
                            DTW / tone / fluency are needed.
        use_tts_reference:  synthesize a reference audio for accuracy DTW
                            and prosody comparison. If False, accuracy &
                            prosody fall back to "no reference" mode.
        asr_engine:         optional ASR engine override, e.g. "aliyun-asr".
        tts_engine:         optional TTS engine override, e.g. "aliyun-tts".
        tts_voice:          optional TTS voice override for engines that use it.
    """
    user_pre = preprocess(user_audio)
    user_mfcc = extract_mfcc(user_pre.wav)
    user_f0, user_times, user_voiced = extract_f0(user_pre.raw_wav)

    user_alignment = align(user_pre.raw_wav, reference_text,
                            vad_segments=user_pre.vad_segments)

    # Reference path (TTS + same feature extraction).  Multiple engines can be
    # passed as "mimo-tts,aliyun-tts" for multi-reference scoring.
    references: list[ReferenceArtifacts] = []
    ref_pre = None
    ref_alignment: list = []
    ref_f0 = ref_times = ref_voiced = None
    ref_audio_path = None
    if use_tts_reference:
        for engine, engine_voice in _parse_tts_engines(tts_engine):
            try:
                references.append(_collect_reference(
                    reference_text, engine=engine, voice=engine_voice or tts_voice,
                ))
            except Exception as e:
                # Network/TTS failure shouldn't kill the whole pipeline.
                print(f"[pipeline] TTS reference unavailable ({engine}): {e!r}")
        if references:
            primary_ref = references[0]
            ref_audio_path = primary_ref.audio_path
            ref_pre = primary_ref.pre
            ref_f0 = primary_ref.f0
            ref_times = primary_ref.f0_times
            ref_voiced = primary_ref.voiced
            ref_alignment = primary_ref.alignment

    # ASR for completeness.
    recognized_text = ""
    if use_asr:
        try:
            recognized_text = recognize(user_pre.raw_wav, engine=asr_engine)
        except Exception as e:
            print(f"[pipeline] ASR unavailable: {e!r}")

    # ------------------ scoring --------------------------------------------
    dim_scores: Dict[str, Any] = {}

    if references:
        accuracy_scores = [
            score_accuracy(
                user_mfcc, ref.mfcc, user_alignment, ref.alignment,
                user_voiced=user_voiced, ref_voiced=ref.voiced,
            )
            for ref in references if ref.alignment and ref.mfcc.size > 0
        ]
        dim_scores["accuracy"] = _best_score(accuracy_scores) if accuracy_scores else None
    else:
        # No reference → score accuracy purely on alignment-coverage heuristic.
        from src.scoring.accuracy import AccuracyScore, SyllableAccuracy
        in_vocab = [s for s in user_alignment if getattr(s, "in_vocab", True)]
        rough = 60.0 + 30.0 * (len(in_vocab) / max(len(user_alignment), 1))
        per = [
            SyllableAccuracy(
                char=s.char, score=rough, dtw_cost=0.0,
                duration=s.duration,
                articulation_score=rough,
            )
            for s in user_alignment
        ]
        dim_scores["accuracy"] = AccuracyScore(overall=rough, per_syllable=per)

    if references:
        tone_scores = [
            score_tone(user_f0, user_voiced, user_alignment,
                       ref.f0, ref.voiced, ref.alignment)
            for ref in references if ref.alignment
        ]
        dim_scores["tone"] = _best_score(tone_scores) if tone_scores else None
    else:
        dim_scores["tone"] = score_tone(user_f0, user_voiced, user_alignment)
    dim_scores["fluency"] = score_fluency(
        user_pre.vad_segments, user_alignment, user_pre.duration,
    )

    if references:
        prosody_scores = [
            score_prosody(user_f0, user_voiced, ref.f0, ref.voiced)
            for ref in references
        ]
        dim_scores["prosody"] = _best_score(prosody_scores, prefer_per_syllable=False)
    else:
        # No reference → grade prosody on user's own pitch range only.
        from src.scoring.prosody import ProsodyScore
        u_voiced = user_f0[user_voiced & (user_f0 > 0)]
        if u_voiced.size >= 5:
            u_st = 12.0 * np.log2(np.maximum(u_voiced, 1e-3) / 100.0)
            rng = float(np.percentile(u_st, 95) - np.percentile(u_st, 5))
            r_score = float(np.clip(100.0 * rng / 10.0, 0.0, 100.0))
        else:
            rng, r_score = 0.0, 0.0
        dim_scores["prosody"] = ProsodyScore(
            overall=r_score, contour_similarity=0.0,
            range_score=r_score, pitch_range_semitones=rng,
        )

    dim_scores["completeness"] = score_completeness(recognized_text, reference_text)
    dim_scores["confidence"] = score_confidence(
        dim_scores,
        user_voiced=user_voiced,
        user_duration=user_pre.duration,
        user_voiced_duration=user_pre.voiced_duration,
        has_tts_reference=bool(references),
        has_asr=bool(use_asr and recognized_text),
    )
    dim_scores = {k: v for k, v in dim_scores.items() if v is not None}

    result = aggregate(dim_scores)
    report = build_report(result,
                          reference_text=reference_text,
                          recognized_text=recognized_text)

    return AssessmentArtifacts(
        report=report,
        user_pre=user_pre,
        ref_pre=ref_pre,
        user_alignment=user_alignment,
        ref_alignment=ref_alignment,
        recognized_text=recognized_text,
        reference_text=reference_text,
        user_f0=user_f0, user_f0_times=user_times, user_voiced=user_voiced,
        ref_f0=ref_f0, ref_f0_times=ref_times, ref_voiced=ref_voiced,
        ref_audio_path=ref_audio_path,
    )
