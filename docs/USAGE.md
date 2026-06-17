# 使用说明

## 一、环境准备

项目使用 `uv` 管理 Python 虚拟环境和依赖，推荐 Python 3.10。

```bash
uv venv --python 3.10
uv sync
```

以后运行命令时统一使用 `uv run`，这样会自动使用项目里的 `.venv`。

## 二、命令行：快速跑通一次评测

```bash
uv run python scripts/smoke_check.py
```

首次运行会：
1. 通过 TTS 合成 *"今天天气真好。"* 的标准发音（约 2 秒，需要网络）。
2. 下载 `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` 模型 (≈1.2 GB)
   到 `data/cache/hf/`（仅第一次）。
3. 用合成的标准发音冒充用户朗读，跑通五维评分。

后续运行命中缓存，几秒内完成。

使用阿里云 Qwen-TTS 做标准音：

```bash
uv sync --extra aliyun
export DASHSCOPE_API_KEY=your_key
uv run python scripts/smoke_check.py --tts-engine aliyun-tts --tts-voice Neil --asr-engine aliyun-asr
```

使用小米 MiMo-V2.5-TTS 做标准音：

```bash
uv sync --extra mimo
export MIMO_API_KEY=your_key
export TTS_ENGINE=mimo-tts
uv run python scripts/smoke_check.py --tts-engine mimo-tts,aliyun-tts --asr-engine aliyun-asr
```

`mimo-tts,aliyun-tts` 会同时生成 MiMo 和 Qwen 两个标准音，逐字选择更合适的参考分数。


## 三、生成错误模拟实验表

```bash
uv run python scripts/benchmark_pronunciation.py --tts-engine mimo-tts,aliyun-tts --asr-engine aliyun-asr
```

脚本会生成 9 个样本：标准音、漏读句尾、中间漏读、局部静音、
中间长停顿、整体慢读、整体降调、局部降调、轻度噪声。
输出文件在 `data/cache/benchmarks/pronunciation_benchmark.md` 和 `.csv`。
这张表可以用来说明：完整度、流利度、声调、韵律会分别响应不同类型的错误。

## 四、启动 Gradio Web UI

```bash
uv run python -m src.app.gradio_app
```

浏览器自动打开 `http://127.0.0.1:7860`。

界面操作：
1. **左侧** — 在"参考文本"中输入或选择示例文本。
2. **左侧** — 录音（点 🎤）或上传 wav 文件。
3. **左侧** — 是否使用 TTS 合成参考音（推荐开启）。
4. 点 **开始评测**。

**右侧输出标签：**
- **得分概览** — 总分 + 五维分。
- **维度得分** — 柱状图直观展示。
- **F0 对比** — 用户 F0 vs 参考 F0，红色高亮声调误读字。
- **波形 / 有声段** — 绿框表示 VAD 检测到的有声片段。
- **频谱** — Mel 频谱图。
- **标准发音参考** — 内置播放器，跟读用。
- **ASR 识别结果** — 完整度评分依据。
- **完整报告** — 含逐字诊断表的 Markdown。
  逐字表里包含整字准确度、声母分、韵母分、发声覆盖、完整度、评分可信度、声调分和声调置信度。

## 五、API 用法

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
print(art.report["confidence"])               # 评分可信度
print(art.report["per_syllable"][0]["initial_score"])  # 第一个字的声母段分数
print(art.report["per_syllable"][0]["final_score"])    # 第一个字的韵母段分数

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

## 六、配置调优

所有阈值集中在 `config.py`，常见调整：

| 参数 | 含义 | 默认 |
|---|---|---|
| `DIM_WEIGHTS` | 五维加权 | acc 0.35 / tone 0.25 / fluency 0.15 / prosody 0.15 / completeness 0.10 |
| `EDGE_TTS_VOICE` | 参考音的 voice 名 | `zh-CN-XiaoxiaoNeural` |
| `ASR_ENGINE` | 完整度识别引擎，`"wav2vec2"`、`"aliyun-asr"` 或 `"auto"` | `"wav2vec2"` |
| `TTS_ENGINE` | `"edge-tts"`、`"aliyun-tts"`、`"mimo-tts"` 或 `"f5-tts"` | `"edge-tts"`，也可以在 `.env` 里改成 `mimo-tts` |
| `ALIYUN_TTS_VOICE` | 阿里云 Qwen-TTS 音色 | `Neil` |
| `MIMO_TTS_VOICE` | 小米 MiMo-V2.5-TTS 音色 | `白桦` |
| `TONE_TEMPLATES` | Chao 五度模板曲线 | 55/35/214/51/neutral |
| `VAD_AGGRESSIVENESS` | webrtcvad 灵敏度 0..3 | 2 |
| `TARGET_SYLLABLES_PER_SEC` | 流利度峰值语速 | 4.0 |

## 七、跑测试

```bash
uv run python -m pytest -q                       # 全部测试
uv run python -m pytest tests/test_scoring.py -v # 单文件
```

## 八、常见问题

**Q1. 第一次跑很慢？**
首次运行需要下载 wav2vec2 (~1.2 GB)、首次合成新参考文本需要联网。模型 / 音频
均缓存到 `data/cache/` 与 `data/standard_audio/`，后续运行非常快。

**Q2. 报错 "Failed to load wav2vec2"？**
若 torch < 2.6 同时 transformers 较新，会触发 CVE-2025-32434 限制。本项目
已加 `use_safetensors=True` 走 safetensors 通道。uv 环境当前会安装较新的
torch，通常可以直接加载模型；若仍然失败，可在 `config.py` 换一个有
safetensors 文件的中文 wav2vec2 模型。

**Q3. 声调评分整体偏低？**
连续语流中实际声调会受句调影响（陈述句末降调、语气词等）。在 PSC 单音节 /
多音节朗读项目里测试时声调评分会更准确。也可调高
`config.DIM_WEIGHTS["accuracy"]` 降低 `tone` 占比。

**Q4. 想换 TTS 引擎？**
改 `config.py`：
```python
TTS_ENGINE = "aliyun-tts"
```
阿里云 Qwen-TTS 需要安装可选依赖并配置 `DASHSCOPE_API_KEY`。MiMo-V2.5-TTS
需要安装可选依赖并配置 `MIMO_API_KEY`。如果想让网页默认走 MiMo，可以在 `.env` 里写
`TTS_ENGINE=mimo-tts`。F5-TTS 仍然保留，它需要一段参考音频
作为 prompt 才能克隆音色。

目前两路参考音都默认使用偏正式、偏教材朗读的风格提示。完整度识别也可以改成 `ASR_ENGINE=aliyun-asr`，用阿里云 Qwen-ASR 提高短句识别稳定性。`--tts-engine mimo-tts,aliyun-tts` 会启用多参考音融合，降低单一 TTS 音色和句调带来的误扣。
