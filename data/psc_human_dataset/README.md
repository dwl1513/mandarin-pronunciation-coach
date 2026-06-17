# 普通话水平测试真人范读数据集

本目录用于存放项目的正式真人标准普通话实验数据。数据集来自普通话水平
测试朗读作品范读材料，用于验证和校准本项目的发音评测算法。

当前规模：

- 朗读作品：50 篇
- 真人范读来源：2 套
- 音频文件：100 条 MP3 + 100 条 16 kHz 单声道 WAV
- ASR 转写：100 条
- 正式基准：50 组真人标准音交叉评测

## 数据来源

- 蚌埠医科大学工会：普通话水平测试朗读作品 50 篇应试辅导  
  https://gh.bbmu.edu.cn/info/1147/3437.htm
- 北京普通话学会：普通话水平测试 2021 年版朗读作品 50 篇连声音档  
  https://www.beijingputonghua.com/psc/ldzp/ldzp.htm

## 目录结构

```text
data/psc_human_dataset/
  manifest.json                 数据集清单：作品、正文、来源、音频路径
  audio/
    bbmu/                       蚌埠医科大学工会真人范读
    beijing_putonghua/          北京普通话学会真人范读
  transcripts/                  ASR 转写结果
    summary.json                100 条转写汇总
  benchmarks/                   真人标准音交叉评测结果
```

音频文件体积较大，并且来源版权需要谨慎处理，所以 `audio/**/*.mp3` 和
`audio/**/*.wav` 默认由 Git 忽略。清单、转写和评测报告可以用于课程报告和
本地演示。

## 生成方式

抓取并转码完整数据集：

```bash
uv run python scripts/collect_psc_reference_audio.py
```

生成 ASR 转写：

```bash
uv run python scripts/transcribe_psc_human_dataset.py --engine aliyun-asr
```

只重试失败条目：

```bash
uv run python scripts/transcribe_psc_human_dataset.py --retry-errors
```

只重试指定作品和来源：

```bash
uv run python scripts/transcribe_psc_human_dataset.py --numbers 18 --source bbmu --retry-errors
```

运行真人标准音交叉评测：

```bash
uv run python scripts/benchmark_human_reference.py --limit 50 --workers 4
```

## 当前 ASR 质量

使用 Qwen-ASR 对 100 条真人范读做转写，结果如下：

- 转写条数：100
- 失败条数：0
- 最低字符覆盖率：95.71%
- 平均字符覆盖率：99.06%
- 最低完整度：95.71
- 平均完整度：99.06

这说明云端 ASR 对这批标准普通话朗读的识别非常稳定，可以作为完整度分析和
文本核验依据。低覆盖样本主要来自文本版本差异或同音近形写法，例如“她/他”、
“幽美/优美”、“刷刷/唰唰”、“地/的”和儿化词差异。

## 当前基准结果

基准方式：用 BBMU 真人范读作为参考音，用北京普通话学会同篇真人范读作为
待测音，做 50 篇交叉评测。

校准后的结果：

- 样本数：50
- 总分均值：94.58
- 总分最低值：93.40
- 声学均分均值：95.13
- 声韵母均值：93.67
- 声调均值：93.14
- 流利度均值：97.70
- 韵律均值：95.99

基准文件：

- `benchmarks/human_reference_benchmark.md`
- `benchmarks/human_reference_benchmark.csv`
- `benchmarks/human_reference_benchmark.json`

## 校准说明

这套数据暴露了两个原算法问题：

- 长篇朗读中，两位标准真人的停连、重音和句调会自然不同，逐字 F0 窗口会
  出现少量错位。声调评分已改为“调类特征 + 参考 F0 趋势证据”的融合方法，
  并把低 F0 覆盖字作为证据不足处理；整体声调分再用真人标准范读数据做
  标尺校准，使标准范读落在优秀区间。
- 普通话水平测试朗读是 400 到 600 字长文，正式范读有大量语义停顿。流利度
  已按长篇朗读场景单独校准，避免把正常断句误判为卡顿。

校准后，两套独立真人标准范读互评稳定在 93 分以上，均值接近 95 分。这样
更符合真人标准范读的直观质量，同时仍保留声学证据、逐字 F0 曲线和明显错误
惩罚能力。
