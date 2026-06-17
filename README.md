# 普通话发音评测系统

本项目是语音处理课程大作业。系统面向普通话水平测试练习场景，用户选择正式朗读片段或输入自定义文本，录音朗读后，系统给出五维评分、逐字诊断、ASR 识别结果、F0 对比图、频谱图、波形图和改进建议。

当前展示应用采用：

- 后端：FastAPI
- 前端：React + Vite + Tailwind
- Python 环境：uv
- 前端包管理：pnpm
- 固定例句：普通话水平测试 PSC 朗读作品片段
- 固定例句标准音：真人范读截取片段
- 自定义文本标准音：MiMo / Qwen / Edge TTS
- ASR：Qwen-ASR 或本地 wav2vec2

## 快速启动

### 1. 准备 Python 环境

项目使用 `uv` 管理 Python 依赖和虚拟环境。

```bash
uv venv --python 3.10
uv sync
```

如果需要使用阿里云 Qwen-ASR / Qwen-TTS：

```bash
uv sync --extra aliyun
```

如果需要使用小米 MiMo-TTS：

```bash
uv sync --extra mimo
```

也可以一次性安装：

```bash
uv sync --extra aliyun --extra mimo
```

### 2. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

然后在 `.env` 里填入自己的 Key。`.env` 已经被 Git 忽略，不能提交到仓库。

常用配置如下：

```env
DASHSCOPE_API_KEY=你的阿里云 DashScope Key
ALIYUN_ASR_MODEL=qwen3-asr-flash
ALIYUN_TTS_MODEL=qwen3-tts-flash
ALIYUN_TTS_VOICE=Cherry
ALIYUN_TTS_LANGUAGE_TYPE=Chinese

MIMO_API_KEY=你的 MiMo Key
MIMO_TTS_MODEL=mimo-v2.5-tts
MIMO_TTS_VOICE=白桦
MIMO_TTS_BASE_URL=https://api.xiaomimimo.com/v1
```

固定 PSC 例句会优先使用本地真人标准音，不需要现场调用 TTS。自定义文本才需要 TTS 生成标准音。

### 3. 启动后端

在项目根目录运行：

```bash
uv run uvicorn src.server.main:app --reload --port 8000
```

后端启动后，可以访问：

```text
http://127.0.0.1:8000/api/health
```

返回 `ok: true` 就说明后端正常。

### 4. 启动前端

打开另一个终端：

```bash
cd frontend
pnpm install
pnpm dev
```

浏览器访问：

```text
http://localhost:5173/
```

前端默认请求：

```text
http://127.0.0.1:8000
```

如果后端地址需要修改，可以在前端环境变量中设置：

```bash
VITE_API_BASE=http://127.0.0.1:8000 pnpm dev
```

## 展示应用怎么用

1. 启动 FastAPI 后端。
2. 启动 React 前端。
3. 打开 `http://localhost:5173/`。
4. 在左侧选择一个 PSC 例句，或输入自定义文本。
5. 点击“播放标准音”，可以听真人标准范读。
6. 点击“开始录音”，按文本朗读。
7. 录音结束后点击“提交评测”。
8. 右侧会显示评测结果。

结果包括：

- 综合得分
- 五维分数
- 改进建议
- 参考文本和 ASR 识别文本对比
- 逐字诊断
- 用户录音和标准音回放
- 用户 F0 与标准 F0 对比图
- 用户录音频谱图
- 用户录音波形和 VAD
- 可信度证据

## 当前固定例句

前端固定例句已经替换为 12 条 PSC 正式朗读片段。

例句配置文件：

```text
frontend/src/data/practice-examples.json
```

当前 12 条例句：

```text
psc-01-beijing-spring      北京的春节
psc-02-spring              春
psc-03-hurry               匆匆
psc-06-nature-language     大自然的语言
psc-08-dinghu-spring       鼎湖山听泉
psc-11-qiantang-tide       观潮
psc-13-summer-seaside      海滨仲夏夜
psc-18-jinci               晋祠
psc-23-mogao               莫高窟
psc-30-hakka-house         世界民居奇葩
psc-41-summer-palace       颐和园
psc-45-taiwan              中国的宝岛台湾
```

这些例句的标准音是从真人普通话范读中人工标注时间戳后截取出来的。前端使用的标准音文件在：

```text
frontend/public/audio/examples/psc-*.wav
```

人工标注时间戳保存在：

```text
data/psc_human_dataset/example_clips.json
```

片段定义保存在：

```text
data/psc_human_dataset/example_clip_specs.json
```

如果修改了时间戳或例句定义，重新生成前端例句和标准音：

```bash
uv run python scripts/generate_psc_example_assets.py
```

## 标准音和 F0 对比

固定 PSC 例句的标准音已经是本地真人截取音频。

当用户选择固定例句并提交评测时，后端会根据 `example_id` 找到对应标准音：

```text
frontend/public/audio/examples/psc-*.wav
```

然后 pipeline 会从该真人标准音中提取：

- 参考 MFCC
- 参考 F0
- 参考 F0 时间轴
- 参考对齐结果

所以前端 F0 图里的参考 F0，就是当前固定例句对应真人标准音的 F0。

自定义文本没有本地真人标准音，系统会调用 TTS 生成标准音，再提取参考 F0。

## 后端接口

主要接口如下：

```text
GET  /api/health
GET  /api/examples
GET  /api/examples/{example_id}/audio
POST /api/assess
GET  /api/assess-jobs/{assessment_id}
GET  /api/results/{assessment_id}/{file_name}
```

评测接口是任务式设计。

提交录音：

```text
POST /api/assess
```

后端会立即返回任务 ID：

```json
{
  "id": "assessment_id",
  "status_url": "/api/assess-jobs/assessment_id"
}
```

前端轮询：

```text
GET /api/assess-jobs/{assessment_id}
```

任务状态包括：

```text
queued
running
done
failed
```

完成后，`result` 字段里会包含完整评测结果。

每次评测生成的声学证据保存在：

```text
data/cache/web_results/{assessment_id}/
```

典型文件：

```text
user.wav
reference.wav
waveform.png
spectrogram.png
f0.png
```

## PSC 片段标注工具

如果需要重新人工截取标准音，可以启动后端后访问：

```text
http://127.0.0.1:8000/tools/clipper
```

这个工具可以：

- 听完整真人原音
- 看波形
- 设置开始时间
- 设置结束时间
- 预听截取片段
- 保存时间戳

保存后会更新：

```text
data/psc_human_dataset/example_clips.json
```

然后重新导出：

```bash
uv run python scripts/generate_psc_example_assets.py
```

## 算法流程

系统评测一条录音时，大致流程如下：

```text
用户录音
  -> 音频预处理
  -> VAD 有声段检测
  -> ASR 识别
  -> 文本完整度评分
  -> 声韵母 MFCC-DTW 评分
  -> F0 提取
  -> 声调评分
  -> 韵律评分
  -> 流利度评分
  -> 五维加权汇总
  -> 逐字诊断和建议
  -> 生成波形、频谱、F0 对比图
```

五维分数：

| 维度 | 含义 |
|---|---|
| 声韵母 | 基于 MFCC-DTW，比对用户录音和标准音的局部声学相似度 |
| 声调 | 基于 F0 曲线、声调轮廓、升降趋势和参考音 F0 |
| 流利度 | 基于语速、停顿、停顿占比和节奏稳定性 |
| 韵律 | 基于整句 F0 走势和音域范围 |
| 完整度 | 基于 ASR 文本和参考文本的字符级覆盖情况 |

## 真人数据集和实验数据

正式真人普通话数据集在：

```text
data/psc_human_dataset/
```

重要文件：

```text
data/psc_human_dataset/manifest.json
data/psc_human_dataset/transcripts/summary.json
data/psc_human_dataset/transcripts/asr_quality_report.md
data/psc_human_dataset/transcripts/asr_quality_report.csv
data/psc_human_dataset/benchmarks/human_reference_benchmark.md
data/psc_human_dataset/benchmarks/human_reference_benchmark.csv
data/psc_human_dataset/benchmarks/human_reference_benchmark.json
```

本地真人完整音频目录：

```text
data/psc_human_dataset/audio/
```

完整音频体积较大，没有提交到 Git。前端用到的 12 条短标准音已经单独截取并保存在：

```text
frontend/public/audio/examples/
```

## 常用脚本

生成前端 PSC 例句和真人标准音：

```bash
uv run python scripts/generate_psc_example_assets.py
```

采集 PSC 真人数据集：

```bash
uv run python scripts/collect_psc_reference_audio.py
```

批量转写 PSC 真人数据：

```bash
uv run python scripts/transcribe_psc_human_dataset.py --engine aliyun-asr
```

审计 ASR 转写质量：

```bash
uv run python scripts/audit_psc_asr_transcripts.py
```

真人标准音交叉评测：

```bash
uv run python scripts/benchmark_human_reference.py --limit 50 --workers 4
```

受控错误模拟实验：

```bash
uv run python scripts/benchmark_pronunciation.py --tts-engine mimo-tts,aliyun-tts --asr-engine aliyun-asr
```

## 测试和检查

运行测试：

```bash
uv run python -m pytest -q
```

检查后端和脚本：

```bash
uv run ruff check src/server/main.py src/server/schemas.py src/server/clipper.py scripts/generate_psc_example_assets.py
```

检查前端构建：

```bash
cd frontend
pnpm build
```

## 项目结构

```text
mandarin-pronunciation-coach/
├── README.md
├── pyproject.toml
├── uv.lock
├── config.py
├── 实验报告.md
├── data/
│   ├── psc_human_dataset/
│   │   ├── manifest.json
│   │   ├── example_clip_specs.json
│   │   ├── example_clips.json
│   │   ├── transcripts/
│   │   └── benchmarks/
│   └── cache/
├── frontend/
│   ├── public/audio/examples/
│   └── src/
│       ├── App.tsx
│       └── data/practice-examples.json
├── scripts/
│   ├── generate_psc_example_assets.py
│   ├── collect_psc_reference_audio.py
│   ├── transcribe_psc_human_dataset.py
│   ├── audit_psc_asr_transcripts.py
│   └── benchmark_human_reference.py
├── src/
│   ├── server/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── examples.py
│   │   ├── render.py
│   │   └── clipper.py
│   ├── audio/
│   ├── asr/
│   ├── features/
│   ├── scoring/
│   ├── feedback/
│   └── pipeline.py
└── tests/
```

## 说明

- `.env` 包含密钥，不能提交。
- `data/cache/` 是运行缓存，不能提交。
- `data/psc_human_dataset/audio/` 是完整真人音频，体积较大，默认不提交。
- `frontend/public/audio/examples/psc-*.wav` 是前端展示用的短标准音，可以提交。
- 固定 PSC 例句使用真人标准音；自定义文本使用 TTS 标准音。

## 许可证

MIT
