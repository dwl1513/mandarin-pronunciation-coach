import {
  Activity,
  AudioLines,
  BookOpen,
  BrainCircuit,
  ChevronRight,
  Check,
  CircleAlert,
  Clock3,
  FileAudio,
  Mic,
  Pause,
  Play,
  RefreshCcw,
  Search,
  Sparkles,
  Waves,
  X,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"

import { Badge } from "./components/ui/badge"
import { Button } from "./components/ui/button"
import { Card, CardContent, CardHeader } from "./components/ui/card"
import { Meter } from "./components/ui/meter"
import { Select } from "./components/ui/select"
import { Textarea } from "./components/ui/textarea"
import practiceExamplesData from "./data/practice-examples.json"
import { cn } from "./lib/utils"

const dimensionLabels: Record<string, string> = {
  accuracy: "声韵母",
  tone: "声调",
  fluency: "流利度",
  prosody: "韵律",
  completeness: "完整度",
}

type PracticeExample = {
  id: string
  title: string
  level: string
  focus: string
  scene: string
  duration: string
  text: string
  tags: string[]
  audioPath: string
}

type DiagnosticRow = {
  char: string
  pinyin: string
  acc: number
  initial: number | null
  final: number
  voiced: number
  completeness: string
  tone: string
  toneScore: number
  confidence: string
  note: string
}

const practiceExamples = practiceExamplesData as PracticeExample[]

const demoReport = {
  overall: 99.8,
  recognized: "今天天气真好我们一起去公园散步吧",
  confidence: {
    overall: 98.3,
    dims: {
      signal: 100,
      reference: 100,
      asr: 100,
      f0: 91.7,
      accuracy: 100,
    },
  },
  dims: {
    accuracy: 100,
    tone: 99.2,
    fluency: 100,
    prosody: 100,
    completeness: 100,
  },
  syllables: [
    {
      char: "今",
      pinyin: "jin",
      acc: 100,
      initial: 100,
      final: 100,
      voiced: 100,
      completeness: "已读",
      tone: "1→1",
      toneScore: 100,
      confidence: "高",
      note: "",
    },
    {
      char: "天",
      pinyin: "tian",
      acc: 100,
      initial: 100,
      final: 100,
      voiced: 100,
      completeness: "已读",
      tone: "1→1",
      toneScore: 100,
      confidence: "高",
      note: "",
    },
    {
      char: "气",
      pinyin: "qi",
      acc: 100,
      initial: 100,
      final: 100,
      voiced: 100,
      completeness: "已读",
      tone: "4→4",
      toneScore: 100,
      confidence: "高",
      note: "",
    },
    {
      char: "我",
      pinyin: "wo",
      acc: 98.9,
      initial: null,
      final: 98.5,
      voiced: 100,
      completeness: "已读",
      tone: "3→0",
      toneScore: 0,
      confidence: "中",
      note: "声调难判",
    },
    {
      char: "们",
      pinyin: "men",
      acc: 88.7,
      initial: 93,
      final: 87.7,
      voiced: 90.9,
      completeness: "漏读",
      tone: "5→0",
      toneScore: 0,
      confidence: "低",
      note: "疑似漏读",
    },
    {
      char: "一",
      pinyin: "yi",
      acc: 55.3,
      initial: null,
      final: 81.9,
      voiced: 30.3,
      completeness: "已读",
      tone: "4→0",
      toneScore: 0,
      confidence: "低",
      note: "有效发声不足",
    },
    {
      char: "起",
      pinyin: "qi",
      acc: 98.6,
      initial: 96.3,
      final: 98.7,
      voiced: 100,
      completeness: "已读",
      tone: "3→2",
      toneScore: 23.6,
      confidence: "中",
      note: "F0 轮廓差异",
    },
  ],
}

const benchmarkRows = [
  { name: "clean", score: 99.4, signal: "标准音上限" },
  { name: "drop_tail", score: 90.2, signal: "句尾漏读" },
  { name: "drop_middle", score: 86.3, signal: "中间漏读" },
  { name: "mute_middle", score: 87.6, signal: "局部静音" },
  { name: "long_pause", score: 94.9, signal: "长停顿" },
  { name: "slow", score: 95.1, signal: "语速偏慢" },
  { name: "pitch_down", score: 85.2, signal: "整体降调" },
  { name: "local_pitch_down", score: 92.7, signal: "局部降调" },
  { name: "noise", score: 93.6, signal: "噪声干扰" },
]

const waveformBars = [
  18, 28, 46, 72, 64, 38, 30, 68, 80, 54, 32, 20, 26, 62, 88, 76, 42, 28,
  18, 22, 52, 70, 48, 30, 24, 60, 82, 74, 36, 22,
]

const pinyinMap: Record<string, string> = {
  今: "jin",
  天: "tian",
  气: "qi",
  真: "zhen",
  好: "hao",
  我: "wo",
  们: "men",
  一: "yi",
  起: "qi",
  去: "qu",
  公: "gong",
  园: "yuan",
  散: "san",
  步: "bu",
  吧: "ba",
  妈: "ma",
  骑: "qi",
  马: "ma",
  慢: "man",
  骂: "ma",
  老: "lao",
  师: "shi",
  让: "rang",
  认: "ren",
  练: "lian",
  习: "xi",
  普: "pu",
  通: "tong",
  话: "hua",
  声: "sheng",
  调: "diao",
  请: "qing",
  清: "qing",
  楚: "chu",
  地: "di",
  区: "qu",
  分: "fen",
  翘: "qiao",
  舌: "she",
  音: "yin",
  和: "he",
  平: "ping",
  想: "xiang",
  买: "mai",
  斤: "jin",
  新: "xin",
  鲜: "xian",
  苹: "ping",
  果: "guo",
  两: "liang",
  瓶: "ping",
  牛: "niu",
  奶: "nai",
  下: "xia",
  午: "wu",
  三: "san",
  点: "dian",
  在: "zai",
  图: "tu",
  书: "shu",
  馆: "guan",
  开: "kai",
  会: "hui",
  把: "ba",
  这: "zhe",
  杯: "bei",
  温: "wen",
  水: "shui",
  放: "fang",
  桌: "zhuo",
  子: "zi",
  右: "you",
  边: "bian",
  北: "bei",
  京: "jing",
  欢: "huan",
  迎: "ying",
  看: "kan",
  升: "sheng",
  旗: "qi",
  春: "chun",
  到: "dao",
  了: "le",
  花: "hua",
  里: "li",
  的: "de",
  小: "xiao",
  树: "shu",
  发: "fa",
  芽: "ya",
  出: "chu",
  租: "zu",
  车: "che",
  司: "si",
  机: "ji",
  已: "yi",
  经: "jing",
  学: "xue",
  校: "xiao",
  门: "men",
  口: "kou",
  等: "deng",
  你: "ni",
  十: "shi",
  首: "shou",
  古: "gu",
  诗: "shi",
  读: "du",
  来: "lai",
  抑: "yi",
  扬: "yang",
  顿: "dun",
  挫: "cuo",
  很: "hen",
  有: "you",
  韵: "yun",
  味: "wei",
  中: "zhong",
  国: "guo",
  人: "ren",
  民: "min",
  热: "re",
  爱: "ai",
  河: "he",
  也: "ye",
  美: "mei",
  生: "sheng",
  活: "huo",
}

const punctuationPattern = /[，。！？、；：,.!?;\s]/u

function App() {
  const [text, setText] = useState(practiceExamples[0].text)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedBenchmark, setSelectedBenchmark] = useState("drop_middle")
  const [selectedExampleId, setSelectedExampleId] = useState(practiceExamples[0].id)
  const [isExampleDrawerOpen, setIsExampleDrawerOpen] = useState(false)

  const selectedExample = useMemo(
    () =>
      practiceExamples.find((item) => item.id === selectedExampleId) ?? {
        id: "custom",
        title: "自定义文本",
        level: "自定义",
        focus: "手动输入",
        scene: "自由练习",
        duration: "--",
        text,
        tags: ["自定义"],
        audioPath: "",
      },
    [selectedExampleId, text],
  )

  const diagnosticRows = useMemo(() => createDiagnosticRows(text), [text])

  const activeBenchmark = useMemo(
    () => benchmarkRows.find((item) => item.name === selectedBenchmark) ?? benchmarkRows[0],
    [selectedBenchmark],
  )

  function runDemo() {
    setIsAnalyzing(true)
    window.setTimeout(() => setIsAnalyzing(false), 900)
  }

  function updateText(value: string) {
    setText(value)
    setSelectedExampleId("custom")
  }

  function selectExample(example: PracticeExample) {
    setSelectedExampleId(example.id)
    setText(example.text)
    setIsExampleDrawerOpen(false)
  }

  return (
    <main className="h-dvh overflow-hidden bg-background text-foreground max-lg:h-auto max-lg:min-h-dvh max-lg:overflow-y-auto">
      <div className="fine-grid pointer-events-none fixed inset-0 opacity-80" />
      <section className="relative mx-auto flex h-dvh w-full max-w-[1500px] flex-col px-3 py-3 lg:px-4 max-lg:h-auto max-lg:min-h-dvh">
        <TopBar />

        <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 py-3 lg:grid-cols-[320px_minmax(0,1fr)_360px]">
          <InputRail
            text={text}
            setText={updateText}
            isAnalyzing={isAnalyzing}
            runDemo={runDemo}
            selectedExample={selectedExample}
            openExamples={() => setIsExampleDrawerOpen(true)}
          />
          <ScoreStage
            diagnosticRows={diagnosticRows}
            isAnalyzing={isAnalyzing}
            recognizedText={normalizeReferenceText(text)}
          />
          <InsightRail
            selectedBenchmark={selectedBenchmark}
            setSelectedBenchmark={setSelectedBenchmark}
            activeBenchmark={activeBenchmark}
            selectedExample={selectedExample}
          />
        </div>
      </section>
      {isExampleDrawerOpen ? (
        <ExampleDrawer
          examples={practiceExamples}
          onClose={() => setIsExampleDrawerOpen(false)}
          onSelect={selectExample}
          selectedExampleId={selectedExample.id}
        />
      ) : null}
    </main>
  )
}

function TopBar() {
  return (
    <header className="panel-border flex h-12 shrink-0 items-center justify-between rounded-panel bg-panel px-3 md:px-4">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex size-8 items-center justify-center rounded-full bg-foreground text-background">
          <AudioLines className="size-4" />
        </div>
        <div className="min-w-0">
          <h1 className="truncate text-sm font-semibold tracking-tight">
            Mandarin Pronunciation Coach
          </h1>
          <p className="hidden text-xs text-muted md:block">
            普通话发音评测工作台
          </p>
        </div>
      </div>
      <div className="hidden items-center gap-2 md:flex">
        <Badge variant="good">算法定版</Badge>
        <Badge variant="outline">56 tests</Badge>
        <Badge variant="outline">MiMo + Qwen</Badge>
      </div>
      <Button variant="ghost" size="sm" className="md:hidden" aria-label="刷新">
        <RefreshCcw />
      </Button>
    </header>
  )
}

type InputRailProps = {
  text: string
  setText: (value: string) => void
  isAnalyzing: boolean
  runDemo: () => void
  selectedExample: PracticeExample
  openExamples: () => void
}

function InputRail({
  text,
  setText,
  isAnalyzing,
  runDemo,
  selectedExample,
  openExamples,
}: InputRailProps) {
  return (
    <aside className="flex min-h-0 flex-col gap-3">
      <Card className="shrink-0">
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between">
            <SectionTitle icon={Mic} title="输入" />
            <div className="flex items-center gap-2">
              <Badge variant="outline">{selectedExample.level}</Badge>
              <Badge variant="outline">16 kHz</Badge>
            </div>
          </div>
          <div className="flex items-start justify-between gap-3 rounded-panel border border-border bg-background p-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">
                {selectedExample.title}
              </p>
              <p className="mt-1 max-h-10 overflow-hidden text-xs leading-5 text-muted-strong">
                {selectedExample.focus}
              </p>
            </div>
            <Button
              className="shrink-0"
              onClick={openExamples}
              size="sm"
              type="button"
              variant="outline"
            >
              <BookOpen />
              例句库
            </Button>
          </div>
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            aria-label="参考文本"
          />
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <UploadTile icon={Mic} label="录音" desc="麦克风输入" />
            <UploadTile icon={FileAudio} label="上传" desc="wav / mp3" />
          </div>
          <Button
            className="w-full"
            size="lg"
            onClick={runDemo}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <Activity className="animate-pulse" />
                分析中
              </>
            ) : (
              <>
                <Sparkles />
                开始评测
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <Card className="min-h-0 flex-1 overflow-hidden">
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between">
            <SectionTitle icon={BrainCircuit} title="引擎" />
            <span className="mono text-xs text-muted">live</span>
          </div>
          <Field label="TTS 标准音">
            <Select defaultValue="mimo-qwen">
              <option value="mimo-qwen">MiMo + Qwen 多参考音</option>
              <option value="mimo">MiMo-V2.5-TTS</option>
              <option value="qwen">Qwen-TTS</option>
            </Select>
          </Field>
          <Field label="ASR 完整度">
            <Select defaultValue="qwen-asr">
              <option value="qwen-asr">Qwen-ASR</option>
              <option value="wav2vec2">本地 wav2vec2</option>
              <option value="auto">自动</option>
            </Select>
          </Field>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2">
          {[
            ["TTS 标准音", "ready"],
            ["Qwen-ASR", "ready"],
            ["F0 提取", "ready"],
            ["MFCC-DTW", "ready"],
          ].map(([label, state]) => (
            <div
              className="rounded-panel border border-border bg-background px-2.5 py-2 text-sm"
              key={label}
            >
              <span className="block truncate text-xs text-muted-strong">{label}</span>
              <span className="mt-1 flex items-center gap-1.5 text-xs text-good">
                <Check className="size-3.5" />
                {state}
              </span>
            </div>
          ))}
        </CardContent>
      </Card>
    </aside>
  )
}

function ScoreStage({
  diagnosticRows,
  isAnalyzing,
  recognizedText,
}: {
  diagnosticRows: DiagnosticRow[]
  isAnalyzing: boolean
  recognizedText: string
}) {
  return (
    <section className="grid min-h-0 grid-rows-[auto_auto_minmax(0,1fr)] gap-3">
      <Card className="relative overflow-hidden">
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="mb-1.5 text-sm text-muted-strong">
                总分
              </p>
              <div className="flex items-end gap-3">
                <span className="mono text-6xl font-semibold leading-none tracking-[-0.03em] md:text-7xl">
                  {demoReport.overall.toFixed(1)}
                </span>
                <span className="pb-2 text-sm text-muted">/ 100</span>
              </div>
            </div>
            <div className="rounded-full border border-border bg-background px-3 py-2 text-right">
              <p className="text-xs text-muted">可信度</p>
              <p className="mono text-xl font-semibold">
                {demoReport.confidence.overall.toFixed(1)}
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 md:grid-cols-5">
            {Object.entries(demoReport.dims).map(([key, value]) => (
              <MetricBlock key={key} label={dimensionLabels[key]} value={value} />
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-3 md:grid-cols-[1.15fr_0.85fr]">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <SectionTitle icon={Waves} title="声学证据" />
            <div className="flex items-center gap-2 text-xs text-muted">
              <span className="size-2 rounded-full bg-foreground" />
              用户
              <span className="size-2 rounded-full border border-foreground/70" />
              标准音
            </div>
          </CardHeader>
          <CardContent>
            <WavePanel isAnalyzing={isAnalyzing} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <SectionTitle icon={Clock3} title="可信度证据" />
            <Badge variant="good">高</Badge>
          </CardHeader>
          <CardContent className="space-y-3">
            {Object.entries(demoReport.confidence.dims).map(([label, value]) => (
              <Meter
                key={label}
                label={confidenceLabel(label)}
                value={value}
                tone={value >= 85 ? "good" : value >= 70 ? "warn" : "bad"}
              />
            ))}
          </CardContent>
        </Card>
      </div>

      <Card className="flex min-h-0 flex-col overflow-hidden">
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <SectionTitle icon={Activity} title="逐字诊断" />
          <span className="hidden max-w-[52%] truncate text-xs text-muted md:block">
            ASR: {recognizedText}
          </span>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 overflow-auto">
          <SyllableTable rows={diagnosticRows} />
        </CardContent>
      </Card>
    </section>
  )
}

function ExampleDrawer({
  examples,
  onClose,
  onSelect,
  selectedExampleId,
}: {
  examples: PracticeExample[]
  onClose: () => void
  onSelect: (example: PracticeExample) => void
  selectedExampleId: string
}) {
  const [query, setQuery] = useState("")
  const filteredExamples = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    if (!keyword) {
      return examples
    }
    return examples.filter((example) =>
      [
        example.title,
        example.level,
        example.focus,
        example.scene,
        example.text,
        ...example.tags,
      ]
        .join(" ")
        .toLowerCase()
        .includes(keyword),
    )
  }, [examples, query])

  return (
    <div className="fixed inset-0 z-40">
      <button
        aria-label="关闭例句库"
        className="absolute inset-0 bg-foreground/10"
        onClick={onClose}
        type="button"
      />
      <aside
        aria-label="例句库"
        aria-modal="true"
        className="panel-border absolute bottom-3 right-3 top-3 flex w-[460px] max-w-[calc(100vw-1.5rem)] flex-col rounded-panel bg-panel shadow-[0_20px_70px_oklch(0.17_0_0_/_0.14)]"
        role="dialog"
      >
        <header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
          <div>
            <h2 className="text-base font-semibold tracking-tight">例句库</h2>
            <p className="mt-1 text-xs text-muted">
              固定评测材料，覆盖声调、韵母、平翘舌和朗读韵律。
            </p>
          </div>
          <Button aria-label="关闭例句库" onClick={onClose} size="icon" variant="ghost">
            <X />
          </Button>
        </header>

        <div className="shrink-0 border-b border-border p-3">
          <label className="relative block">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted" />
            <input
              className="h-10 w-full rounded-full border border-border bg-background pl-9 pr-3 text-sm text-foreground outline-none transition duration-200 placeholder:text-muted focus:border-border-strong focus:bg-panel"
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索：声调、平翘舌、长句..."
              type="search"
              value={query}
            />
          </label>
        </div>

        <div className="min-h-0 flex-1 space-y-2 overflow-auto p-3">
          {filteredExamples.map((example) => {
            const selected = example.id === selectedExampleId
            return (
              <button
                className={cn(
                  "w-full rounded-panel border p-3 text-left transition duration-200",
                  selected
                    ? "border-foreground bg-foreground text-background"
                    : "border-border bg-background text-foreground hover:border-border-strong hover:bg-panel-raised",
                )}
                key={example.id}
                onClick={() => onSelect(example)}
                type="button"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold">{example.title}</span>
                      <span
                        className={cn(
                          "rounded-full border px-2 py-0.5 text-xs",
                          selected
                            ? "border-background/25 text-background/75"
                            : "border-border text-muted-strong",
                        )}
                      >
                        {example.level}
                      </span>
                      <span
                        className={cn(
                          "mono text-xs",
                          selected ? "text-background/60" : "text-muted",
                        )}
                      >
                        {example.duration}
                      </span>
                    </div>
                    <p
                      className={cn(
                        "mt-2 text-sm leading-6",
                        selected ? "text-background/82" : "text-muted-strong",
                      )}
                    >
                      {example.text}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {example.tags.map((tag) => (
                        <span
                          className={cn(
                            "rounded-full px-2 py-1 text-xs",
                            selected
                              ? "bg-background/12 text-background/72"
                              : "bg-panel text-muted-strong",
                          )}
                          key={tag}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                  <ChevronRight className="mt-1 size-4 shrink-0" />
                </div>
              </button>
            )
          })}
        </div>
      </aside>
    </div>
  )
}

type InsightRailProps = {
  selectedBenchmark: string
  setSelectedBenchmark: (value: string) => void
  activeBenchmark: (typeof benchmarkRows)[number]
  selectedExample: PracticeExample
}

function InsightRail({
  selectedBenchmark,
  setSelectedBenchmark,
  activeBenchmark,
  selectedExample,
}: InsightRailProps) {
  const [playback, setPlayback] = useState<{
    path: string
    status: "idle" | "playing" | "error"
  }>({ path: "", status: "idle" })
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const hasStandardAudio = Boolean(selectedExample.audioPath)
  const audioStatus =
    playback.path === selectedExample.audioPath ? playback.status : "idle"

  useEffect(() => {
    audioRef.current?.pause()
    audioRef.current = null
  }, [selectedExample.audioPath])

  useEffect(() => {
    return () => {
      audioRef.current?.pause()
    }
  }, [])

  function playStandardAudio() {
    if (!hasStandardAudio) {
      setPlayback({ path: selectedExample.audioPath, status: "error" })
      return
    }

    audioRef.current?.pause()
    setPlayback({ path: selectedExample.audioPath, status: "idle" })
    const audio = new Audio(selectedExample.audioPath)
    audioRef.current = audio
    audio.addEventListener(
      "ended",
      () => setPlayback({ path: selectedExample.audioPath, status: "idle" }),
      { once: true },
    )
    audio.addEventListener(
      "error",
      () => setPlayback({ path: selectedExample.audioPath, status: "error" }),
      { once: true },
    )
    setPlayback({ path: selectedExample.audioPath, status: "playing" })
    audio
      .play()
      .catch(() => setPlayback({ path: selectedExample.audioPath, status: "error" }))
  }

  return (
    <aside className="flex min-h-0 flex-col gap-3">
      <Card className="shrink-0">
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <SectionTitle icon={Play} title="标准发音" />
          <Button
            disabled={!hasStandardAudio}
            onClick={playStandardAudio}
            size="sm"
            type="button"
            variant="subtle"
          >
            <Play />
            {audioStatus === "playing" ? "播放中" : "播放"}
          </Button>
        </CardHeader>
        <CardContent>
          <div className="flex h-12 items-center gap-1 rounded-panel border border-border bg-background px-3">
            {waveformBars.slice(0, 24).map((height, index) => (
              <div
                className="flex flex-1 items-center"
                key={`${height}-${index}`}
              >
                <span
                  className="block w-full rounded-full bg-foreground"
                  style={{ height: `${Math.max(6, height * 0.42)}px` }}
                />
              </div>
            ))}
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-muted">
            <span className="truncate">
              {selectedExample.title} · mimo-v2.5-tts · 白桦
            </span>
            <span className="mono shrink-0">{selectedExample.duration}</span>
          </div>
          {audioStatus === "error" ? (
            <p className="mt-2 text-xs leading-5 text-bad">
              音频还没生成或加载失败，请运行生成脚本后再播放。
            </p>
          ) : null}
        </CardContent>
      </Card>

      <Card className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between">
            <SectionTitle icon={CircleAlert} title="错误模拟实验" />
            <Badge variant="outline">9 samples</Badge>
          </div>
          <p className="text-sm leading-5 text-muted-strong">
            用同一段标准音构造受控错误，展示不同评分维度如何响应。
          </p>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 space-y-2 overflow-auto">
          {benchmarkRows.map((row) => (
            <button
              className={cn(
                "w-full rounded-panel border px-3 py-2.5 text-left transition duration-200",
                row.name === selectedBenchmark
                  ? "border-foreground bg-foreground text-background"
                  : "border-border bg-background text-foreground hover:border-border-strong hover:bg-panel-raised",
              )}
              key={row.name}
              onClick={() => setSelectedBenchmark(row.name)}
              type="button"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="mono text-sm">{row.name}</span>
                <span className="mono text-sm">{row.score.toFixed(1)}</span>
              </div>
              <p
                className={cn(
                  "mt-1 text-xs",
                  row.name === selectedBenchmark ? "text-background/70" : "text-muted",
                )}
              >
                {row.signal}
              </p>
            </button>
          ))}
        </CardContent>
      </Card>

      <Card className="shrink-0">
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between">
            <SectionTitle icon={Pause} title="当前样本" />
            <Badge
              variant={
                activeBenchmark.score >= 95
                  ? "good"
                  : activeBenchmark.score >= 90
                    ? "warn"
                    : "bad"
              }
            >
              {activeBenchmark.score.toFixed(1)}
            </Badge>
          </div>
          <div className="rounded-panel border border-border bg-background p-3">
            <p className="mono text-sm">{activeBenchmark.name}</p>
            <p className="mt-2 text-sm leading-5 text-muted-strong">
              {activeBenchmark.signal} 样本用于答辩展示，可解释完整度、声调和流利度的分工。
            </p>
          </div>
        </CardHeader>
      </Card>
    </aside>
  )
}

function UploadTile({
  icon: Icon,
  label,
  desc,
}: {
  icon: typeof Mic
  label: string
  desc: string
}) {
  return (
    <button
      className="rounded-panel border border-border bg-background p-2.5 text-left transition duration-200 hover:border-border-strong hover:bg-panel-raised"
      type="button"
    >
      <Icon className="mb-3 size-5 text-foreground" />
      <p className="text-sm font-medium">{label}</p>
      <p className="mt-1 text-xs text-muted">{desc}</p>
    </button>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="grid gap-2">
      <span className="text-xs text-muted-strong">{label}</span>
      {children}
    </label>
  )
}

function SectionTitle({
  icon: Icon,
  title,
}: {
  icon: typeof AudioLines
  title: string
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="size-4 text-muted-strong" />
      <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
    </div>
  )
}

function MetricBlock({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-panel border border-border bg-background p-2.5">
      <p className="text-xs text-muted">{label}</p>
      <p className="mono mt-1.5 text-2xl font-semibold">{value.toFixed(1)}</p>
    </div>
  )
}

function WavePanel({ isAnalyzing }: { isAnalyzing: boolean }) {
  return (
    <div className="relative h-32 overflow-hidden rounded-panel border border-border bg-background px-4 py-4">
      <div className="absolute inset-x-4 top-1/2 h-px bg-border" />
      <div className="relative flex h-full items-center gap-1">
        {waveformBars.map((height, index) => (
          <div className="flex flex-1 items-center justify-center" key={index}>
            <span
              className={cn(
                "block w-full max-w-2 rounded-full bg-foreground/90 transition-all duration-500",
                isAnalyzing && "opacity-60",
              )}
              style={{ height: `${height}%` }}
            />
          </div>
        ))}
      </div>
      <svg
        aria-hidden="true"
        className="absolute inset-x-4 bottom-4 h-16 w-[calc(100%-2rem)]"
        viewBox="0 0 560 110"
        preserveAspectRatio="none"
      >
        <path
          d="M0 68 C40 48 54 24 94 40 C140 58 144 82 188 72 C232 62 222 28 270 30 C324 32 314 88 365 78 C410 70 414 34 458 42 C500 50 512 76 560 58"
          fill="none"
          stroke="oklch(0.17 0 0 / 0.72)"
          strokeWidth="2"
        />
        <path
          d="M0 74 C42 62 62 40 100 48 C138 56 150 70 190 65 C232 60 240 36 280 42 C322 48 330 82 372 74 C420 65 425 48 468 52 C512 56 524 70 560 62"
          fill="none"
          stroke="oklch(0.17 0 0 / 0.28)"
          strokeDasharray="7 8"
          strokeWidth="2"
        />
      </svg>
    </div>
  )
}

function SyllableTable({ rows }: { rows: DiagnosticRow[] }) {
  return (
    <div className="min-w-[760px] overflow-hidden rounded-panel border border-border">
      <div className="grid grid-cols-[52px_76px_repeat(7,minmax(76px,1fr))_1.4fr] border-b border-border bg-panel-raised px-3 py-2 text-xs text-muted-strong">
        <span>字</span>
        <span>拼音</span>
        <span>准确度</span>
        <span>声母</span>
        <span>韵母</span>
        <span>发声</span>
        <span>完整度</span>
        <span>声调</span>
        <span>可信度</span>
        <span>备注</span>
      </div>
      {rows.map((item, index) => (
        <div
          className="grid grid-cols-[52px_76px_repeat(7,minmax(76px,1fr))_1.4fr] items-center border-b border-border/70 px-3 py-2 text-sm last:border-b-0"
          key={`${item.char}-${item.pinyin}-${index}`}
        >
          <span className="text-lg font-semibold">{item.char}</span>
          <span className="mono text-muted-strong">{item.pinyin}</span>
          <ScoreCell value={item.acc} />
          <ScoreCell value={item.initial} />
          <ScoreCell value={item.final} />
          <ScoreCell value={item.voiced} />
          <Badge variant={item.completeness === "已读" ? "good" : "bad"}>
            {item.completeness}
          </Badge>
          <span className="mono">{item.tone}</span>
          <Badge
            variant={
              item.confidence === "高"
                ? "good"
                : item.confidence === "中"
                  ? "warn"
                  : "bad"
            }
          >
            {item.confidence}
          </Badge>
          <span className="truncate text-muted-strong">{item.note || "正常"}</span>
        </div>
      ))}
    </div>
  )
}

function ScoreCell({ value }: { value: number | null }) {
  if (value === null) {
    return <span className="text-muted">-</span>
  }
  const tone = value >= 90 ? "text-good" : value >= 70 ? "text-warn" : "text-bad"
  return <span className={cn("mono", tone)}>{value.toFixed(1)}</span>
}

function confidenceLabel(key: string) {
  const labels: Record<string, string> = {
    signal: "有效语音",
    reference: "标准音参考",
    asr: "ASR 覆盖",
    f0: "F0 可用性",
    accuracy: "准确度依据",
  }
  return labels[key] ?? key
}

function normalizeReferenceText(value: string) {
  return value.replace(/[，。！？、；：,.!?;\s]/gu, "")
}

function createDiagnosticRows(value: string): DiagnosticRow[] {
  const chars = Array.from(value).filter((char) => !punctuationPattern.test(char))
  return chars.map((char, index) => {
    const seed = char.charCodeAt(0) + index * 17
    const weak = seed % 11 === 0
    const medium = seed % 7 === 0
    const toneWeak = seed % 13 === 0
    const acc = weak ? 82.4 : medium ? 93.6 : 99.2
    const final = weak ? 84.8 : medium ? 94.1 : 99.4
    const voiced = weak ? 86.2 : 100
    const completeness = seed % 19 === 0 ? "漏读" : "已读"
    const confidence = weak ? "中" : completeness === "漏读" ? "低" : "高"

    return {
      char,
      pinyin: pinyinMap[char] ?? "-",
      acc,
      initial: char === "一" || char === "儿" ? null : weak ? 85.5 : 99.1,
      final,
      voiced,
      completeness,
      tone: toneWeak ? "3→2" : `${(seed % 4) + 1}→${(seed % 4) + 1}`,
      toneScore: toneWeak ? 72.5 : 99.1,
      confidence,
      note:
        completeness === "漏读"
          ? "疑似漏读"
          : toneWeak
            ? "声调略偏"
            : weak
              ? "韵母收尾偏弱"
              : "正常",
    }
  })
}

export default App
