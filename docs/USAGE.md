# 使用说明

## 一、环境准备

项目默认运行在 conda 虚拟环境 **`f5tts`**（已含 torch 2.5.1+cu121、
torchaudio、transformers、librosa、scikit-learn、pypinyin、pytest 等核心
依赖）。

```powershell
conda activate f5tts
pip install -r requirements.txt
```

> 项目 README 列出了不使用 `f5tts` 时的备选环境创建方法。

## 二、命令行：快速跑通一次评测

```powershell
python scripts/smoke_check.py
```

首次运行会：
1. 通过 edge-tts 合成 *"今天天气真好。"* 的标准发音（约 2 秒，需要网络）。
2. 下载 `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` 模型 (≈1.2 GB)
   到 `data/cache/hf/`（仅第一次）。
3. 用合成的标准发音冒充用户朗读，跑通五维评分。

后续运行命中缓存，几秒内完成。

## 三、启动 Gradio Web UI

```powershell
python -m src.app.gradio_app
```

浏览器自动打开 `http://127.0.0.1:7860`。

界面操作：
1. **左侧** — 在"参考文本"中输入或选择示例文本。
2. **左侧** — 录音（点 🎤）或上传 wav 文件。
3. **左侧** — 是否使用 TTS 合成参考音（推荐开启）。
4. 点 **开始评测**。

**右侧输出标签：**
- **得分概览** — 总分 + 五维分。
- **维度雷达** — 柱状图直观展示。
- **F0 对比** — 用户 F0 vs 参考 F0，红色高亮声调误读字。
- **波形 / 有声段** — 绿框表示 VAD 检测到的有声片段。
- **频谱** — Mel 频谱图。
- **标准发音参考** — 内置播放器，跟读用。
- **ASR 识别结果** — 完整度评分依据。
- **完整报告** — 含逐字诊断表的 Markdown。

## 四、API 用法

```python
from src.pipeline import assess

art = assess(
    user_audio="my_recording.wav",            # 或 (sr, np.ndarray) / numpy 数组
    reference_text="今天天气真好。",
    use_asr=True,                             # 是否跑 ASR (识别 → 完整度)
    use_tts_reference=True,                   # 是否生成 TTS 参考音 (准确度 / 韵律)
)

print(art.report["overall"])                  # 总分
print(art.report["dims"])                     # 5 维分
print(art.report["markdown"])                 # 完整 markdown 报告

# 进一步可视化
from src.feedback.visualize import plot_f0_comparison
fig = plot_f0_comparison(
    art.user_f0, art.user_f0_times,
    art.ref_f0,  art.ref_f0_times,
    alignment=art.user_alignment,
    per_syllable=art.report["per_syllable"],
)
fig.savefig("f0.png")
```

## 五、配置调优

所有阈值集中在 `config.py`，常见调整：

| 参数 | 含义 | 默认 |
|---|---|---|
| `DIM_WEIGHTS` | 五维加权 | acc 0.35 / tone 0.25 / fluency 0.15 / prosody 0.15 / completeness 0.10 |
| `EDGE_TTS_VOICE` | 参考音的 voice 名 | `zh-CN-XiaoxiaoNeural` |
| `TTS_ENGINE` | `"edge-tts"` 或 `"f5-tts"` | `"edge-tts"` |
| `TONE_TEMPLATES` | Chao 五度模板曲线 | 55/35/214/51/neutral |
| `VAD_AGGRESSIVENESS` | webrtcvad 灵敏度 0..3 | 2 |
| `TARGET_SYLLABLES_PER_SEC` | 流利度峰值语速 | 4.0 |

## 六、跑测试

```powershell
python -m pytest -q                       # 36 个测试，~3.5 分钟
python -m pytest tests/test_scoring.py -v # 单文件
```

> ⚠️ Windows 上务必用 `python -m pytest` 而不是裸 `pytest`。如果多个 conda 环
> 境都装了 pytest，PATH 里 `pytest.exe` 可能解析到别的环境，导致激活了
> `f5tts` 但实际跑的是另一个 env 的 Python，从而出现 `ModuleNotFoundError:
> No module named 'librosa' / 'pypinyin'`。`python -m pytest` 一定用当前激
> 活环境的 Python。

## 七、常见问题

**Q1. 第一次跑很慢？**
首次运行需要下载 wav2vec2 (~1.2 GB)、首次合成新参考文本需要联网。模型 / 音频
均缓存到 `data/cache/` 与 `data/standard_audio/`，后续运行非常快。

**Q2. 报错 "Failed to load wav2vec2"？**
若 torch < 2.6 同时 transformers 较新，会触发 CVE-2025-32434 限制。本项目
已加 `use_safetensors=True` 走 safetensors 通道；若仍然失败可：
- 升级 torch (`pip install torch==2.6.* torchaudio==2.6.*`)，但可能与 F5-TTS 不兼容；
- 或在 `config.py` 换一个有 safetensors 文件的中文 wav2vec2 模型。

**Q3. 声调评分整体偏低？**
连续语流中实际声调会受句调影响（陈述句末降调、语气词等）。在 PSC 单音节 /
多音节朗读项目里测试时声调评分会更准确。也可调高
`config.DIM_WEIGHTS["accuracy"]` 降低 `tone` 占比。

**Q4. 想换 TTS 引擎？**
改 `config.py`：
```python
TTS_ENGINE = "f5-tts"
```
F5-TTS 需要一段参考音频作为 prompt 才能克隆音色。请见
`src/feedback/tts.py:_f5_tts` 的参数。
