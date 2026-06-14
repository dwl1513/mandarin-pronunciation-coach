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
from src.scoring.fluency import score_fluency
from src.scoring.prosody import score_prosody
from src.scoring.tone import score_tone


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


def assess(user_audio: Union[str, Path, np.ndarray, tuple],
           reference_text: str,
           *,
           use_asr: bool = True,
           use_tts_reference: bool = True) -> AssessmentArtifacts:
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
    """
    user_pre = preprocess(user_audio)
    user_mfcc = extract_mfcc(user_pre.wav)
    user_f0, user_times, user_voiced = extract_f0(user_pre.raw_wav)

    user_alignment = align(user_pre.raw_wav, reference_text,
                            vad_segments=user_pre.vad_segments)

    # Reference path (TTS + same feature extraction).
    ref_pre = None
    ref_mfcc = np.zeros((0, user_mfcc.shape[1] if user_mfcc.ndim == 2 else 1),
                        dtype=np.float32)
    ref_alignment: list = []
    ref_f0 = ref_times = ref_voiced = None
    ref_audio_path = None
    if use_tts_reference:
        try:
            ref_audio_path = synth_reference(reference_text)
            ref_pre = preprocess(ref_audio_path)
            ref_mfcc = extract_mfcc(ref_pre.wav)
            ref_f0, ref_times, ref_voiced = extract_f0(ref_pre.raw_wav)
            ref_alignment = align(ref_pre.raw_wav, reference_text,
                                   vad_segments=ref_pre.vad_segments)
        except Exception as e:
            # Network/TTS failure shouldn't kill the whole pipeline.
            print(f"[pipeline] TTS reference unavailable: {e!r}")
            ref_pre = None
            ref_alignment = []

    # ASR for completeness.
    recognized_text = ""
    if use_asr:
        try:
            recognized_text = recognize(user_pre.raw_wav)
        except Exception as e:
            print(f"[pipeline] ASR unavailable: {e!r}")

    # ------------------ scoring --------------------------------------------
    dim_scores: Dict[str, Any] = {}

    if ref_alignment and ref_mfcc.size > 0:
        dim_scores["accuracy"] = score_accuracy(
            user_mfcc, ref_mfcc, user_alignment, ref_alignment,
        )
    else:
        # No reference → score accuracy purely on alignment-coverage heuristic.
        from src.scoring.accuracy import AccuracyScore, SyllableAccuracy
        in_vocab = [s for s in user_alignment if getattr(s, "in_vocab", True)]
        rough = 60.0 + 30.0 * (len(in_vocab) / max(len(user_alignment), 1))
        per = [SyllableAccuracy(char=s.char, score=rough, dtw_cost=0.0,
                                duration=s.duration) for s in user_alignment]
        dim_scores["accuracy"] = AccuracyScore(overall=rough, per_syllable=per)

    dim_scores["tone"] = score_tone(user_f0, user_voiced, user_alignment)
    dim_scores["fluency"] = score_fluency(
        user_pre.vad_segments, user_alignment, user_pre.duration,
    )

    if ref_f0 is not None and ref_voiced is not None:
        dim_scores["prosody"] = score_prosody(
            user_f0, user_voiced, ref_f0, ref_voiced,
        )
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
