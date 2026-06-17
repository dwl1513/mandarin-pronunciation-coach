"""评测结果文件和声学证据图生成。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.audio.capture import save_audio
from src.feedback.visualize import plot_f0_comparison, plot_spectrogram, plot_waveform
from src.pipeline import AssessmentArtifacts


def save_assessment_assets(artifacts: AssessmentArtifacts,
                           result_dir: Path) -> dict[str, str | None]:
    """把算法产物保存成前端可直接展示的音频和图片。"""
    result_dir.mkdir(parents=True, exist_ok=True)

    user_audio = result_dir / "user.wav"
    save_audio(artifacts.user_pre.raw_wav, user_audio, artifacts.user_pre.sr)

    reference_audio: Path | None = None
    if artifacts.ref_pre is not None:
        reference_audio = result_dir / "reference.wav"
        save_audio(artifacts.ref_pre.raw_wav, reference_audio, artifacts.ref_pre.sr)

    waveform = result_dir / "waveform.png"
    spectrogram = result_dir / "spectrogram.png"
    f0_plot = result_dir / "f0.png"

    _save_figure(
        plot_waveform(
            artifacts.user_pre.raw_wav,
            artifacts.user_pre.sr,
            artifacts.user_pre.vad_segments,
            title="用户录音波形与 VAD",
        ),
        waveform,
    )
    _save_figure(
        plot_spectrogram(
            artifacts.user_pre.raw_wav,
            artifacts.user_pre.sr,
            title="用户录音频谱图",
        ),
        spectrogram,
    )
    _save_figure(
        plot_f0_comparison(
            artifacts.user_f0,
            artifacts.user_f0_times,
            artifacts.ref_f0,
            artifacts.ref_f0_times,
            alignment=artifacts.user_alignment,
            per_syllable=artifacts.report.get("per_syllable", []),
            title="用户 F0 与标准音 F0 对比",
        ),
        f0_plot,
    )

    return {
        "user_audio": user_audio.name,
        "reference_audio": None if reference_audio is None else reference_audio.name,
        "waveform": waveform.name,
        "spectrogram": spectrogram.name,
        "f0": f0_plot.name,
    }


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

