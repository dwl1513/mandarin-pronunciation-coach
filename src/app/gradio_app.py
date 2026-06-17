"""Gradio UI — end-to-end demo.

Run with:
    uv run python -m src.app.gradio_app

Layout:
    左侧 — 输入：参考文本 + 麦克风 / 文件上传
    右侧 — 结果：总分 + 维度得分 + F0 对比图 + 逐字诊断表 + 报告 markdown
                  + "标准发音参考"音频播放器
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/app/gradio_app.py` direct execution from the repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr                  # noqa: E402

from src.feedback.visualize import (plot_f0_comparison, plot_score_bars,   # noqa: E402
                                     plot_spectrogram, plot_waveform)
from src.pipeline import assess                                            # noqa: E402

DEFAULT_TEXT = "今天天气真好，我们一起去公园散步吧。"

EXAMPLES = [
    "你好，我叫小明。",
    "今天天气真好，我们一起去公园散步吧。",
    "学而时习之，不亦说乎？",
    "春眠不觉晓，处处闻啼鸟。",
]


def _run(audio, reference_text, use_tts):
    if audio is None:
        return ("请先录音或上传音频。", None, None, None, None, None,
                "", "")
    if not reference_text or not reference_text.strip():
        return ("请输入参考文本。", None, None, None, None, None, "", "")

    art = assess(audio, reference_text.strip(),
                 use_asr=True, use_tts_reference=use_tts)

    score_md = (
        f"### 总分 **{art.report['overall']:.1f}** / 100\n\n"
        + "  ".join(f"- {k}: {v:.1f}" for k, v in art.report["dims"].items())
    )

    fig_bars = plot_score_bars(art.report["dims"])
    fig_wave = plot_waveform(art.user_pre.raw_wav, art.user_pre.sr,
                              vad_segments=art.user_pre.vad_segments,
                              title="用户波形 (绿框=有声段)")
    fig_spec = plot_spectrogram(art.user_pre.raw_wav, art.user_pre.sr,
                                 title="用户频谱")
    fig_f0 = plot_f0_comparison(
        art.user_f0, art.user_f0_times,
        art.ref_f0, art.ref_f0_times,
        alignment=art.user_alignment,
        per_syllable=art.report["per_syllable"],
        title="F0 对比 (蓝=你，橙=参考；红色=声调误读)",
    )

    ref_audio = str(art.ref_audio_path) if art.ref_audio_path else None
    return (score_md, fig_bars, fig_wave, fig_spec, fig_f0,
            ref_audio, art.report["markdown"], art.recognized_text)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="普通话 AI 发音教练", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 🗣️ 普通话 AI 发音教练\n"
            "朗读给定文本 → 系统自动评测**准确度 / 声调 / 流利度 / 韵律 / 完整度**，"
            "给出逐字诊断和标准发音参考。\n"
        )
        with gr.Row():
            with gr.Column(scale=1):
                ref_text = gr.Textbox(
                    label="参考文本",
                    value=DEFAULT_TEXT, lines=3, max_lines=8,
                )
                gr.Examples(EXAMPLES, inputs=ref_text)
                audio = gr.Audio(
                    label="录音 / 上传音频",
                    sources=["microphone", "upload"], type="numpy",
                )
                use_tts = gr.Checkbox(
                    label="使用 TTS 合成标准发音作为参考 (推荐开启)",
                    value=True,
                )
                run_btn = gr.Button("开始评测", variant="primary")
            with gr.Column(scale=2):
                score_md = gr.Markdown(label="得分概览")
                with gr.Tabs():
                    with gr.Tab("维度得分"):
                        plot_bars = gr.Plot()
                    with gr.Tab("F0 对比"):
                        plot_f0 = gr.Plot()
                    with gr.Tab("波形 / 有声段"):
                        plot_wav = gr.Plot()
                    with gr.Tab("频谱"):
                        plot_spec = gr.Plot()
                ref_audio_player = gr.Audio(label="标准发音参考", interactive=False)
                rec_text = gr.Textbox(label="ASR 识别结果", interactive=False)
                report_md = gr.Markdown(label="完整报告")

        run_btn.click(
            _run,
            inputs=[audio, ref_text, use_tts],
            outputs=[score_md, plot_bars, plot_wav, plot_spec, plot_f0,
                     ref_audio_player, report_md, rec_text],
        )
    return app


if __name__ == "__main__":
    build_app().launch()
