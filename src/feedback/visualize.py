"""Matplotlib visualizations for the report (M5).

Each function returns a `matplotlib.figure.Figure` so the Gradio UI can hand
it to `gr.Plot` directly, and tests can call `.savefig()` without GUI back-
ends getting in the way.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")               # safe default for headless / CI
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

# Enable Chinese characters in matplotlib labels (Windows: Microsoft YaHei).
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_waveform(wav: np.ndarray, sr: int,
                  vad_segments: Optional[List[Tuple[float, float]]] = None,
                  title: str = "波形"):
    fig, ax = plt.subplots(figsize=(9, 2.5))
    t = np.arange(len(wav)) / sr
    ax.plot(t, wav, linewidth=0.6, color="#2b6cb0")
    if vad_segments:
        for s, e in vad_segments:
            ax.axvspan(s, e, alpha=0.15, color="#48bb78")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅度")
    ax.set_title(title)
    ax.set_xlim(0, max(t.max() if t.size else 1.0, 0.1))
    fig.tight_layout()
    return fig


def plot_spectrogram(wav: np.ndarray, sr: int, title: str = "频谱图"):
    import librosa
    import librosa.display as ld

    fig, ax = plt.subplots(figsize=(9, 3.0))
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(wav, n_fft=512, hop_length=160)),
        ref=np.max,
    )
    img = ld.specshow(S, sr=sr, hop_length=160, x_axis="time",
                       y_axis="hz", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_f0_comparison(user_f0: np.ndarray, user_times: np.ndarray,
                       ref_f0: Optional[np.ndarray] = None,
                       ref_times: Optional[np.ndarray] = None,
                       alignment: Optional[list] = None,
                       per_syllable: Optional[list] = None,
                       title: str = "F0 对比"):
    """Draw user F0 vs reference F0, mark mispronounced syllables in red."""
    fig, ax = plt.subplots(figsize=(9, 3.0))
    u_voiced = user_f0 > 0
    ax.plot(user_times[u_voiced], user_f0[u_voiced], "o", markersize=3,
            color="#3182ce", label="用户 F0")
    if ref_f0 is not None and ref_times is not None:
        r_voiced = ref_f0 > 0
        ax.plot(ref_times[r_voiced], ref_f0[r_voiced], "x", markersize=4,
                color="#dd6b20", alpha=0.6, label="参考 F0")

    if alignment and per_syllable:
        n = min(len(alignment), len(per_syllable))
        for i in range(n):
            syl = alignment[i]
            info = per_syllable[i]
            mid = (syl.start + syl.end) / 2
            color = "#c53030" if not info.get("tone_ok", True) else "#2f855a"
            ax.text(mid, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 400,
                    info["char"], color=color, ha="center", fontsize=10)
            if not info.get("tone_ok", True):
                ax.axvspan(syl.start, syl.end, color="#fed7d7", alpha=0.35)

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_score_bars(dims: dict, title: str = "维度得分"):
    fig, ax = plt.subplots(figsize=(6, 3.0))
    labels = list(dims.keys())
    values = [dims[k] for k in labels]
    cn_labels = {
        "accuracy": "准确度", "tone": "声调", "fluency": "流利度",
        "prosody": "韵律", "completeness": "完整度",
    }
    show = [cn_labels.get(k, k) for k in labels]
    bars = ax.bar(show, values, color="#4299e1")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("得分")
    ax.set_title(title)
    fig.tight_layout()
    return fig
