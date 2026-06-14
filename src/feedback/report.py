"""Report builder (M5).

Takes a `ScoreResult` and produces both a markdown string for human reading
and a flat JSON-able dict for the Gradio UI / downstream consumers.
"""
from __future__ import annotations

from typing import Dict


def _grade(score: float) -> str:
    if score >= 90:
        return "一甲 (≥97 难达, 90+ 已优秀)"
    if score >= 80:
        return "二甲"
    if score >= 70:
        return "二乙"
    if score >= 60:
        return "三甲"
    return "三乙 / 待加强"


def build_report(result, reference_text: str = "",
                 recognized_text: str = "") -> Dict:
    """Build a {markdown, json} report from a ScoreResult."""
    lines = []
    lines.append(f"# 朗读评测报告\n")
    lines.append(f"**总分:** {result.overall:.1f} / 100  \n")
    lines.append(f"**估算等级:** {_grade(result.overall)}\n")
    lines.append(f"")
    lines.append(f"## 维度得分")
    cn = {"accuracy": "声韵母准确度", "tone": "声调", "fluency": "流利度",
          "prosody": "韵律自然度", "completeness": "完整度"}
    for dim, score in result.dims.items():
        lines.append(f"- **{cn.get(dim, dim)}**: {score:.1f}")
    lines.append("")

    if reference_text:
        lines.append("## 参考文本")
        lines.append(f"> {reference_text}")
        lines.append("")
    if recognized_text:
        lines.append("## 识别结果")
        lines.append(f"> {recognized_text}")
        lines.append("")

    if result.per_syllable:
        lines.append("## 逐字诊断 (高亮 = 误读 / 声调误)")
        grid = []
        for s in result.per_syllable:
            ok = s.get("tone_ok", True) and s.get("acc_score", 0) >= 60
            mark = "✅" if ok else "❌"
            grid.append(
                f"| {s['char']} | {s.get('expected_tone', '?')} → "
                f"{s.get('detected_tone', '?')} | "
                f"{s.get('acc_score', 0):.0f} | "
                f"{s.get('tone_score', 0):.0f} | {mark} | "
                f"{s.get('note', '')} |"
            )
        lines.append("| 字 | 应/实 声调 | 准确度 | 声调分 | 状态 | 备注 |")
        lines.append("|---|---|---|---|---|---|")
        lines.extend(grid)
        lines.append("")

    if result.fluency_detail:
        d = result.fluency_detail
        lines.append("## 流利度细节")
        lines.append(f"- 语速：约 {d.get('speech_rate', 0):.2f} 字/秒")
        lines.append(f"- 停顿次数：{d.get('pause_count', 0)}")
        lines.append(f"- 停顿总时长：{d.get('pause_total', 0):.2f} s")
        lines.append("")

    if result.notes:
        lines.append("## 提示")
        for note in result.notes:
            lines.append(f"- {note}")
        lines.append("")

    markdown = "\n".join(lines)
    return {
        "markdown": markdown,
        "overall": result.overall,
        "dims": result.dims,
        "per_syllable": result.per_syllable,
        "fluency_detail": result.fluency_detail,
        "notes": result.notes,
    }
