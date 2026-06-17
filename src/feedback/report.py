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
    lines.append("# 朗读评测报告\n")
    lines.append(f"**总分:** {result.overall:.1f} / 100  \n")
    lines.append(f"**估算等级:** {_grade(result.overall)}\n")
    lines.append("")
    lines.append("## 维度得分")
    cn = {"accuracy": "声韵母准确度", "tone": "声调", "fluency": "流利度",
          "prosody": "韵律自然度", "completeness": "完整度"}
    for dim, score in result.dims.items():
        lines.append(f"- **{cn.get(dim, dim)}**: {score:.1f}")
    lines.append("")

    if result.confidence:
        lines.append("## 评分可信度")
        confidence = result.confidence
        lines.append(f"- **总体可信度**: {confidence.get('overall', 0):.1f} / 100")
        cn_conf = {
            "signal": "有效语音",
            "reference": "标准音参考",
            "asr": "ASR 覆盖",
            "f0": "F0 可用性",
            "accuracy": "准确度依据",
        }
        for name, score in confidence.get("dims", {}).items():
            lines.append(f"- **{cn_conf.get(name, name)}**: {score:.1f}")
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
            lexical = s.get("lexical_tone", s.get("expected_tone", "?"))
            expected = s.get("expected_tone", "?")
            rule = s.get("tone_rule", "")
            expected_show = (
                f"{lexical}→{expected}（{rule}）"
                if rule and lexical != expected else str(expected)
            )
            time_show = f"{s.get('start', 0):.2f}-{s.get('end', 0):.2f}"
            ref_similarity = s.get("ref_similarity")
            ref_show = "" if ref_similarity is None else f"{ref_similarity:.0f}"
            initial_score = s.get("initial_score")
            final_score = s.get("final_score")
            articulation_score = s.get("articulation_score")
            initial_show = "" if initial_score is None else f"{initial_score:.0f}"
            final_show = "" if final_score is None else f"{final_score:.0f}"
            articulation_show = (
                "" if articulation_score is None else f"{articulation_score:.0f}"
            )
            completeness_show = "已读" if s.get("completeness_ok", True) else "漏读"
            confidence_show = s.get("confidence_level", "")
            grid.append(
                f"| {s['char']} | {s.get('pinyin', '')} | "
                f"{expected_show} | {s.get('detected_tone', '?')} | "
                f"{s.get('acc_score', 0):.0f} | "
                f"{initial_show} | {final_show} | "
                f"{articulation_show} | "
                f"{completeness_show} | "
                f"{confidence_show} | "
                f"{s.get('tone_score', 0):.0f} | {ref_show} | "
                f"{s.get('tone_confidence', 0):.2f} | {time_show} | {mark} | "
                f"{s.get('note', '')} |"
            )
        lines.append("| 字 | 拼音 | 参考声调 | 检测声调 | 准确度 | 声母分 | 韵母分 | 发声覆盖 | 完整度 | 可信度 | 声调分 | 参考相似度 | 置信度 | 时间(s) | 状态 | 备注 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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
        "confidence": result.confidence,
        "notes": result.notes,
    }
