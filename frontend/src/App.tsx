import {
  Activity,
  AudioLines,
  BookOpen,
  Check,
  CircleAlert,
  Clock3,
  FileAudio,
  Mic,
  Play,
  RefreshCcw,
  Search,
  Send,
  SlidersHorizontal,
  Square,
  Volume2,
  Waves,
  X,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"

import { Badge } from "./components/ui/badge"
import { Button } from "./components/ui/button"
import { Card, CardContent, CardHeader } from "./components/ui/card"
import { Lightbox, ZoomableImage } from "./components/ui/lightbox"
import { Meter } from "./components/ui/meter"
import { Select } from "./components/ui/select"
import { Textarea } from "./components/ui/textarea"
import { LiveWaveform } from "./components/ui/waveform"
import practiceExamplesData from "./data/practice-examples.json"
import { api, resolveAsset } from "./lib/api"
import { cn } from "./lib/utils"

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

type AdviceItem = {
  level: "good" | "warn" | "bad"
  title: string
  detail: string
}

type EvidenceLinks = {
  user_audio_url: string
  reference_audio_url: string | null
  waveform_url: string
  spectrogram_url: string
  f0_url: string
}

type AssessmentResponse = {
  id: string
  reference_text: string
  recognized_text: string
  overall: number
  dims: Record<string, number>
  confidence: {
    overall?: number
    dims?: Record<string, number>
    notes?: string[]
  }
  fluency_detail: Record<string, number>
  notes: string[]
  per_syllable: SyllableRow[]
  advice: AdviceItem[]
  evidence: EvidenceLinks
  markdown: string
}

type AssessmentJobCreated = {
  id: string
  status_url: string
}

type AssessmentJobStatus = {
  id: string
  status: "queued" | "running" | "done" | "failed"
  stage: string
  message: string
  progress: number
  error?: string | null
  result?: AssessmentResponse | null
}

type SyllableRow = {
  char: string
  pinyin?: string
  acc_score?: number
  initial_score?: number | null
  final_score?: number | null
  articulation_score?: number | null
  completeness_ok?: boolean
  expected_tone?: number | string
  detected_tone?: number | string
  tone_score?: number
  confidence_level?: string
  note?: string
  start?: number
  end?: number
}

type HealthResponse = {
  ok: boolean
  service: string
  asr_model: string
  tts_model: string
}

type RecorderState = "idle" | "recording" | "recorded"
type EvidenceTab = "f0" | "spectrogram" | "waveform" | "report"

const practiceExamples = practiceExamplesData as PracticeExample[]

const dimensionLabels: Record<string, string> = {
  accuracy: "声韵母",
  tone: "声调",
  fluency: "流利度",
  prosody: "韵律",
  completeness: "完整度",
}

const confidenceLabels: Record<string, string> = {
  signal: "有效语音",
  reference: "标准音参考",
  asr: "ASR 覆盖",
  f0: "F0 可用性",
  accuracy: "准确度依据",
}

function App() {
  const [examples, setExamples] = useState<PracticeExample[]>(practiceExamples)
  const [selectedExampleId, setSelectedExampleId] = useState(practiceExamples[0]?.id ?? "custom")
  const [text, setText] = useState(practiceExamples[0]?.text ?? "")
  const [isExampleDrawerOpen, setIsExampleDrawerOpen] = useState(false)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState("")
  const [asrEngine, setAsrEngine] = useState("aliyun-asr")
  const [ttsEngine, setTtsEngine] = useState("mimo-tts,aliyun-tts")
  const [result, setResult] = useState<AssessmentResponse | null>(null)
  const [isAssessing, setIsAssessing] = useState(false)
  const [assessmentJob, setAssessmentJob] = useState<AssessmentJobStatus | null>(null)
  const [error, setError] = useState("")

  const selectedExample = useMemo(
    () =>
      examples.find((item) => item.id === selectedExampleId) ?? {
        id: "custom",
        title: "自定义文本",
        level: "自定义",
        focus: "手动输入朗读材料",
        scene: "自由练习",
        duration: "--",
        text,
        tags: ["自定义"],
        audioPath: "",
      },
    [examples, selectedExampleId, text],
  )

  const recorder = useRecorder()

  useEffect(() => {
    void loadHealth()
    void loadExamples()
  }, [])

  async function loadHealth() {
    try {
      setHealthError("")
      const response = await fetch(api("/api/health"))
      if (!response.ok) {
        throw new Error("后端未就绪")
      }
      setHealth((await response.json()) as HealthResponse)
    } catch {
      setHealth(null)
      setHealthError("FastAPI 未连接")
    }
  }

  async function loadExamples() {
    try {
      const response = await fetch(api("/api/examples"))
      if (!response.ok) {
        return
      }
      const data = (await response.json()) as PracticeExample[]
      if (Array.isArray(data) && data.length > 0) {
        setExamples(data)
      }
    } catch {
      setExamples(practiceExamples)
    }
  }

  function selectExample(example: PracticeExample) {
    setSelectedExampleId(example.id)
    setText(example.text)
    setResult(null)
    setAssessmentJob(null)
    setError("")
    setIsExampleDrawerOpen(false)
  }

  function updateText(value: string) {
    setText(value)
    setSelectedExampleId("custom")
    setResult(null)
    setAssessmentJob(null)
  }

  async function submitAssessment() {
    if (!recorder.audioBlob) {
      setError("请先录一段朗读音频")
      return
    }
    if (!text.trim()) {
      setError("请先选择例句或输入朗读文本")
      return
    }

    const formData = new FormData()
    formData.append("audio", recorder.audioBlob, recorder.fileName)
    formData.append("reference_text", text.trim())
    formData.append("example_id", selectedExample.id)
    formData.append("asr_engine", asrEngine)
    formData.append("tts_engine", ttsEngine)

    setIsAssessing(true)
    setAssessmentJob(null)
    setResult(null)
    setError("")
    try {
      const response = await fetch(api("/api/assess"), {
        method: "POST",
        body: formData,
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload?.detail || "评测失败")
      }
      const created = payload as AssessmentJobCreated
      const finalStatus = await pollAssessmentJob(created.status_url)
      if (!finalStatus.result) {
        throw new Error("评测完成，但后端没有返回结果")
      }
      setResult(finalStatus.result)
    } catch (err) {
      setResult(null)
      setError(err instanceof Error ? err.message : "评测失败，请重试")
    } finally {
      setIsAssessing(false)
    }
  }

  async function pollAssessmentJob(statusUrl: string): Promise<AssessmentJobStatus> {
    while (true) {
      await delay(900)
      const response = await fetch(api(statusUrl))
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload?.detail || "无法获取评测进度")
      }
      const status = payload as AssessmentJobStatus
      setAssessmentJob(status)
      if (status.status === "done") {
        return status
      }
      if (status.status === "failed") {
        throw new Error(status.error || status.message || "评测失败")
      }
    }
  }

  return (
    <main className="min-h-dvh bg-background text-foreground">
      <div className="fine-grid pointer-events-none fixed inset-0 opacity-70" />

      <div className="relative">
        <TopBar health={health} healthError={healthError} onRefresh={loadHealth} />

        <div className="mx-auto w-full max-w-[1640px] px-4 py-5 lg:px-6">
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-[400px_minmax(0,1fr)]">
            {/* 左栏：练习材料 / 录音 / 设置，随页面滚动时吸顶跟随 */}
            <aside className="flex flex-col gap-5 lg:sticky lg:top-[76px] lg:self-start">
              <PracticePanel
                asrEngine={asrEngine}
                error={error}
                isAssessing={isAssessing}
                onOpenExamples={() => setIsExampleDrawerOpen(true)}
                onSubmit={submitAssessment}
                onTextChange={updateText}
                recorder={recorder}
                selectedExample={selectedExample}
                setAsrEngine={setAsrEngine}
                setTtsEngine={setTtsEngine}
                text={text}
                ttsEngine={ttsEngine}
              />
            </aside>

            {/* 右侧主区：总分 → 建议/识别 → 逐字诊断 → 声学证据 → 可信度，整页铺开 */}
            <div className="min-w-0">
              <ResultPanel
                assessmentJob={assessmentJob}
                isAssessing={isAssessing}
                recorder={recorder}
                result={result}
                selectedExample={selectedExample}
                text={text}
              />
            </div>
          </div>
        </div>
      </div>

      {isExampleDrawerOpen ? (
        <ExampleDrawer
          examples={examples}
          onClose={() => setIsExampleDrawerOpen(false)}
          onSelect={selectExample}
          selectedExampleId={selectedExample.id}
        />
      ) : null}
    </main>
  )
}

function TopBar({
  health,
  healthError,
  onRefresh,
}: {
  health: HealthResponse | null
  healthError: string
  onRefresh: () => void
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-panel/85 backdrop-blur-md">
      <div className="mx-auto flex h-[60px] w-full max-w-[1640px] items-center justify-between px-4 lg:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-full bg-foreground text-background">
            <AudioLines className="size-4.5" />
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

        <div className="flex items-center gap-2">
          <Badge variant={health?.ok ? "good" : "bad"}>
            {health?.ok ? "FastAPI 已连接" : healthError || "后端检查中"}
          </Badge>
          <Badge className="hidden md:inline-flex" variant="outline">
            {health?.asr_model || "Qwen-ASR"}
          </Badge>
          <Badge className="hidden md:inline-flex" variant="outline">
            {health?.tts_model || "MiMo TTS"}
          </Badge>
          <Button
            aria-label="刷新后端状态"
            onClick={onRefresh}
            size="icon"
            type="button"
            variant="ghost"
          >
            <RefreshCcw />
          </Button>
        </div>
      </div>
    </header>
  )
}

type RecorderController = {
  audioBlob: Blob | null
  audioUrl: string
  elapsedMs: number
  error: string
  fileName: string
  reset: () => void
  start: () => Promise<void>
  state: RecorderState
  stop: () => void
  stream: MediaStream | null
}

function PracticePanel({
  asrEngine,
  error,
  isAssessing,
  onOpenExamples,
  onSubmit,
  onTextChange,
  recorder,
  selectedExample,
  setAsrEngine,
  setTtsEngine,
  text,
  ttsEngine,
}: {
  asrEngine: string
  error: string
  isAssessing: boolean
  onOpenExamples: () => void
  onSubmit: () => void
  onTextChange: (value: string) => void
  recorder: RecorderController
  selectedExample: PracticeExample
  setAsrEngine: (value: string) => void
  setTtsEngine: (value: string) => void
  text: string
  ttsEngine: string
}) {
  const [isSettingsOpen, setIsSettingsOpen] = useState(true)
  const canSubmit = recorder.state === "recorded" && text.trim().length > 0 && !isAssessing

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <SectionTitle icon={BookOpen} title="朗读材料" />
            <Button onClick={onOpenExamples} size="sm" type="button" variant="outline">
              <BookOpen />
              例句库
            </Button>
          </div>

          <div className="rounded-panel border border-border bg-background p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold">{selectedExample.title}</p>
                <p className="mt-1 text-xs leading-5 text-muted-strong">
                  {selectedExample.focus}
                </p>
              </div>
              <Badge className="shrink-0" variant="outline">
                {selectedExample.level}
              </Badge>
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {selectedExample.tags.map((tag) => (
                <span
                  className="rounded-full bg-panel px-2 py-1 text-xs text-muted-strong"
                  key={tag}
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          <Textarea
            aria-label="朗读文本"
            className="h-28 text-[15px]"
            onChange={(event) => onTextChange(event.target.value)}
            value={text}
          />
        </CardHeader>
      </Card>

      <Card>
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between">
            <SectionTitle icon={Mic} title="录音朗读" />
            <span className="mono rounded-full border border-border bg-background px-2.5 py-1 text-xs text-muted-strong">
              {formatElapsed(recorder.elapsedMs)}
            </span>
          </div>

          <LiveWaveform
            blob={recorder.audioBlob}
            className="h-28"
            state={recorder.state}
            stream={recorder.stream}
          />

          <div className="grid grid-cols-[1fr_auto] gap-2">
            {recorder.state === "recording" ? (
              <Button onClick={recorder.stop} size="lg" type="button" variant="default">
                <Square />
                结束录音
              </Button>
            ) : (
              <Button
                disabled={isAssessing}
                onClick={() => void recorder.start()}
                size="lg"
                type="button"
                variant="default"
              >
                <Mic />
                {recorder.state === "recorded" ? "重新录音" : "开始录音"}
              </Button>
            )}

            <Button
              disabled={recorder.state === "idle" || isAssessing}
              onClick={recorder.reset}
              size="lg"
              type="button"
              variant="outline"
            >
              <RefreshCcw />
              重置
            </Button>
          </div>

          {recorder.audioUrl ? (
            <audio
              className="h-10 w-full"
              controls
              src={recorder.audioUrl}
            />
          ) : (
            <p className="rounded-panel border border-dashed border-border bg-background px-3 py-2 text-xs leading-5 text-muted-strong">
              点击开始录音后朗读上方文本。录音结束后可以试听，再提交评测。
            </p>
          )}

          {recorder.error ? (
            <InlineMessage tone="bad" text={recorder.error} />
          ) : null}
          {error ? <InlineMessage tone="bad" text={error} /> : null}

          <Button
            className="w-full"
            disabled={!canSubmit}
            onClick={onSubmit}
            size="lg"
            type="button"
          >
            {isAssessing ? (
              <>
                <Activity className="animate-pulse" />
                正在评测
              </>
            ) : (
              <>
                <Send />
                提交评测
              </>
            )}
          </Button>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader className="space-y-3">
          <button
            className="flex w-full items-center justify-between gap-3 text-left"
            onClick={() => setIsSettingsOpen((value) => !value)}
            type="button"
          >
            <SectionTitle icon={SlidersHorizontal} title="评测设置" />
            <Badge variant="outline">{isSettingsOpen ? "收起" : "展开"}</Badge>
          </button>

          <div className={cn("space-y-3", !isSettingsOpen && "hidden")}>
            <Field label="ASR 完整度">
              <Select
                onChange={(event) => setAsrEngine(event.target.value)}
                value={asrEngine}
              >
                <option value="aliyun-asr">Qwen-ASR</option>
                <option value="wav2vec2">本地 wav2vec2</option>
                <option value="auto">自动</option>
              </Select>
            </Field>
            <Field label="TTS 标准音">
              <Select
                onChange={(event) => setTtsEngine(event.target.value)}
                value={ttsEngine}
              >
                <option value="mimo-tts,aliyun-tts">MiMo + Qwen 多参考音</option>
                <option value="mimo-tts">MiMo-V2.5-TTS</option>
                <option value="aliyun-tts">Qwen-TTS</option>
                <option value="edge-tts">Edge-TTS</option>
              </Select>
            </Field>

            <div className="grid grid-cols-2 gap-2 pt-1">
              {["ASR", "TTS", "F0", "DTW"].map((item) => (
                <div
                  className="rounded-panel border border-border bg-background px-3 py-2"
                  key={item}
                >
                  <span className="block text-xs text-muted">{item}</span>
                  <span className="mt-1 flex items-center gap-1.5 text-xs text-good">
                    <Check className="size-3.5" />
                    已接入
                  </span>
                </div>
              ))}
            </div>
          </div>
        </CardHeader>
      </Card>
    </div>
  )
}

function ResultPanel({
  assessmentJob,
  isAssessing,
  recorder,
  result,
  selectedExample,
  text,
}: {
  assessmentJob: AssessmentJobStatus | null
  isAssessing: boolean
  recorder: RecorderController
  result: AssessmentResponse | null
  selectedExample: PracticeExample
  text: string
}) {
  if (isAssessing) {
    return <AssessingPanel job={assessmentJob} />
  }

  if (!result) {
    return (
      <div className="flex flex-col gap-5">
        <Card>
          <CardHeader className="space-y-4">
            <SectionTitle icon={AudioLines} title="本次任务" />
            <div className="rounded-panel border border-border bg-background p-5">
              <p className="text-2xl font-semibold leading-10 tracking-tight md:text-3xl">
                {text || "请选择例句或输入朗读文本"}
              </p>
            </div>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="space-y-3">
            <SectionTitle icon={CircleAlert} title="评测结果" />
            <div className="rounded-panel border border-dashed border-border bg-background p-8 text-center">
              <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-panel-raised">
                <Mic className="size-6 text-muted-strong" />
              </div>
              <p className="mt-4 text-base font-semibold">录音提交后，这里展示完整评测结果。</p>
              <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-muted-strong">
                结果包括总分、五维分、逐字诊断、ASR 对比和声学证据。现场演示时，只需要选一句、录一遍、提交评测。
              </p>
            </div>
          </CardHeader>
        </Card>
      </div>
    )
  }

  const weakRows = result.per_syllable.filter((item) => {
    const acc = Number(item.acc_score ?? 100)
    const tone = Number(item.tone_score ?? 100)
    return item.note || acc < 85 || tone < 80 || item.completeness_ok === false
  })

  return (
    <div className="flex flex-col gap-5">
      <ScoreHero result={result} />

      <div className="grid gap-5 xl:grid-cols-[1fr_0.85fr]">
        <Card>
          <CardHeader className="space-y-3">
            <SectionTitle icon={CircleAlert} title="改进建议" />
            <div className="space-y-2">
              {result.advice.map((item, index) => (
                <AdviceCard item={item} key={`${item.title}-${index}`} />
              ))}
            </div>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="space-y-3">
            <SectionTitle icon={FileAudio} title="识别对比" />
            <CompareBlock label="参考文本" value={result.reference_text} />
            <CompareBlock
              label="ASR 识别"
              value={result.recognized_text || "未返回识别文本"}
            />
          </CardHeader>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <SectionTitle icon={Activity} title="逐字诊断" />
          <Badge variant={weakRows.length ? "warn" : "good"}>
            {weakRows.length ? `${weakRows.length} 个需关注` : "整体正常"}
          </Badge>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <SyllableTable rows={result.per_syllable} />
        </CardContent>
      </Card>

      <EvidenceSection
        key={result.id}
        recorder={recorder}
        result={result}
        selectedExample={selectedExample}
      />
    </div>
  )
}

function ScoreHero({ result }: { result: AssessmentResponse }) {
  const overallTone = scoreTone(result.overall)
  return (
    <Card className="card-shadow-lg overflow-hidden">
      <div className="grid gap-0 lg:grid-cols-[300px_minmax(0,1fr)]">
        {/* 左：总分 + 可信度 */}
        <div className="flex flex-col justify-center gap-4 border-b border-border bg-panel-raised/60 p-6 lg:border-b-0 lg:border-r">
          <div>
            <p className="mb-2 text-sm font-medium text-muted-strong">综合得分</p>
            <div className="flex items-end gap-2">
              <span
                className={cn(
                  "mono text-7xl font-semibold leading-none tracking-[-0.04em]",
                  overallTone === "good" && "text-good",
                  overallTone === "warn" && "text-warn",
                  overallTone === "bad" && "text-bad",
                )}
              >
                {result.overall.toFixed(1)}
              </span>
              <span className="pb-2 text-sm text-muted">/ 100</span>
            </div>
          </div>
          <div className="flex items-center justify-between rounded-panel border border-border bg-background px-4 py-3">
            <div>
              <p className="text-xs text-muted">综合可信度</p>
              <p className="mono mt-0.5 text-2xl font-semibold">
                {(result.confidence.overall ?? 0).toFixed(1)}
              </p>
            </div>
            <Badge variant={scoreTone(result.confidence.overall ?? 0) === "good" ? "good" : "warn"}>
              {(result.confidence.overall ?? 0) >= 85 ? "结果可靠" : "供参考"}
            </Badge>
          </div>
        </div>

        {/* 右：五维分 */}
        <div className="p-6">
          <p className="mb-4 text-sm font-medium text-muted-strong">五维明细</p>
          <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2">
            {Object.entries(result.dims).map(([key, value]) => (
              <div key={key} className="space-y-2">
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-medium">
                    {dimensionLabels[key] ?? key}
                  </span>
                  <span
                    className={cn("mono text-xl font-semibold", scoreText(value))}
                  >
                    {value.toFixed(1)}
                  </span>
                </div>
                <Meter tone={scoreTone(value)} value={value} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  )
}

function EvidenceSection({
  recorder,
  result,
  selectedExample,
}: {
  recorder: RecorderController
  result: AssessmentResponse
  selectedExample: PracticeExample
}) {
  const [tab, setTab] = useState<EvidenceTab>("f0")
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [standardStatus, setStandardStatus] = useState<"idle" | "playing" | "error">("idle")

  useEffect(() => {
    return () => {
      audioRef.current?.pause()
    }
  }, [])

  function playStandardAudio() {
    const src = resolveAsset(result.evidence.reference_audio_url) || selectedExample.audioPath
    if (!src) {
      setStandardStatus("error")
      return
    }
    audioRef.current?.pause()
    const audio = new Audio(src)
    audioRef.current = audio
    setStandardStatus("playing")
    audio.addEventListener("ended", () => setStandardStatus("idle"), { once: true })
    audio.addEventListener("error", () => setStandardStatus("error"), { once: true })
    audio.play().catch(() => setStandardStatus("error"))
  }

  const imageMap: Record<Exclude<EvidenceTab, "report">, string> = {
    f0: resolveAsset(result.evidence.f0_url),
    spectrogram: resolveAsset(result.evidence.spectrogram_url),
    waveform: resolveAsset(result.evidence.waveform_url),
  }
  const labelMap: Record<Exclude<EvidenceTab, "report">, string> = {
    f0: "F0 对比图",
    spectrogram: "频谱图",
    waveform: "波形与 VAD",
  }
  const captionMap: Record<Exclude<EvidenceTab, "report">, string> = {
    f0: "蓝色为用户 F0，橙色为标准音 F0，红色区域表示声调可疑片段。",
    spectrogram: "频谱图展示能量在时间和频率上的分布，可用于说明声学证据。",
    waveform: "绿色区域表示 VAD 检测到的有效语音片段。",
  }

  return (
    <>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        {/* 声学证据大图 */}
        <Card>
          <CardHeader className="space-y-3">
            <div className="flex items-center justify-between">
              <SectionTitle icon={Waves} title="声学证据" />
              <Badge variant="good">已生成</Badge>
            </div>

            <div className="flex flex-wrap gap-1 rounded-full border border-border bg-background p-1">
              {(
                [
                  ["f0", "F0 对比"],
                  ["spectrogram", "频谱图"],
                  ["waveform", "波形 VAD"],
                  ["report", "完整报告"],
                ] as [EvidenceTab, string][]
              ).map(([value, label]) => (
                <button
                  className={cn(
                    "h-9 flex-1 rounded-full px-3 text-sm font-medium transition duration-200",
                    tab === value
                      ? "bg-foreground text-background"
                      : "text-muted-strong hover:bg-panel-raised hover:text-foreground",
                  )}
                  key={value}
                  onClick={() => setTab(value)}
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
          </CardHeader>

          <CardContent>
            {tab === "report" ? (
              <pre className="max-h-[640px] overflow-auto whitespace-pre-wrap rounded-panel border border-border bg-background p-4 text-xs leading-6 text-muted-strong">
                {result.markdown}
              </pre>
            ) : (
              <figure className="space-y-2">
                <ZoomableImage
                  alt={labelMap[tab]}
                  onZoom={() => setLightboxOpen(true)}
                  src={imageMap[tab]}
                />
                <figcaption className="text-xs leading-5 text-muted-strong">
                  {captionMap[tab]}
                </figcaption>
              </figure>
            )}
          </CardContent>
        </Card>

        {/* 右侧：音频回放 + 可信度证据 */}
        <div className="flex flex-col gap-5">
          <Card>
            <CardHeader className="space-y-3">
              <div className="flex items-center justify-between">
                <SectionTitle icon={Volume2} title="音频回放" />
                <Badge variant="outline">{selectedExample.duration}</Badge>
              </div>
              <Button
                disabled={!selectedExample.audioPath && !result.evidence.reference_audio_url}
                onClick={playStandardAudio}
                size="sm"
                type="button"
                variant="outline"
              >
                <Play />
                {standardStatus === "playing" ? "标准音播放中" : "播放标准音"}
              </Button>
              {standardStatus === "error" ? (
                <InlineMessage tone="bad" text="标准音加载失败，请检查后端或例句音频文件。" />
              ) : null}
              <div className="space-y-1.5">
                <p className="text-xs text-muted">我的录音</p>
                {recorder.audioUrl ? (
                  <audio className="h-10 w-full" controls src={recorder.audioUrl} />
                ) : (
                  <p className="rounded-panel border border-dashed border-border bg-background px-3 py-2 text-xs leading-5 text-muted-strong">
                    用户录音会在这里回放。
                  </p>
                )}
              </div>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader className="space-y-3">
              <SectionTitle icon={Clock3} title="可信度证据" />
              {result.confidence.dims ? (
                <div className="space-y-3.5">
                  {Object.entries(result.confidence.dims).map(([key, value]) => (
                    <Meter
                      key={key}
                      label={confidenceLabels[key] ?? key}
                      tone={scoreTone(Number(value))}
                      value={Number(value)}
                    />
                  ))}
                </div>
              ) : (
                <p className="rounded-panel border border-dashed border-border bg-background px-3 py-2 text-xs leading-5 text-muted-strong">
                  评测完成后显示有效语音、标准音参考、ASR 覆盖和 F0 可用性。
                </p>
              )}
            </CardHeader>
          </Card>
        </div>
      </div>

      {lightboxOpen && tab !== "report" ? (
        <Lightbox
          alt={labelMap[tab]}
          caption={captionMap[tab]}
          onClose={() => setLightboxOpen(false)}
          src={imageMap[tab]}
        />
      ) : null}
    </>
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
        className="panel-border absolute bottom-3 right-3 top-3 flex w-[470px] max-w-[calc(100vw-1.5rem)] flex-col rounded-panel bg-panel shadow-[0_20px_70px_oklch(0.17_0_0_/_0.14)]"
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
              placeholder="搜索：声调、平翘舌、长句"
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
                </div>
              </button>
            )
          })}
        </div>
      </aside>
    </div>
  )
}

function useRecorder(): RecorderController {
  const [state, setState] = useState<RecorderState>("idle")
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState("")
  const [elapsedMs, setElapsedMs] = useState(0)
  const [error, setError] = useState("")
  const [fileName, setFileName] = useState("recording.webm")
  const [liveStream, setLiveStream] = useState<MediaStream | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<number | null>(null)
  const startedAtRef = useRef(0)

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
      }
    }
  }, [audioUrl])

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
      }
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  async function start() {
    try {
      cleanup()
      setError("")
      setAudioBlob(null)
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
        setAudioUrl("")
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("当前浏览器没有麦克风录音能力")
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      setLiveStream(stream)
      chunksRef.current = []

      const mimeType = pickMimeType()
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      mediaRecorderRef.current = recorder
      setFileName(`recording.${mimeType.includes("mp4") ? "mp4" : "webm"}`)

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: mimeType || "audio/webm",
        })
        setAudioBlob(blob)
        setAudioUrl(URL.createObjectURL(blob))
        setState("recorded")
        cleanupStream()
      }

      recorder.start()
      setElapsedMs(0)
      setState("recording")
      startTimer()
    } catch (err) {
      cleanup()
      setState("idle")
      setError(err instanceof Error ? err.message : "无法开始录音")
    }
  }

  function stop() {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop()
    }
    stopTimer()
  }

  function reset() {
    cleanup()
    setState("idle")
    setAudioBlob(null)
    setElapsedMs(0)
    setError("")
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl)
      setAudioUrl("")
    }
  }

  function startTimer() {
    stopTimer()
    startedAtRef.current = 0
    timerRef.current = window.setInterval(() => {
      const now = Date.now()
      if (!startedAtRef.current) {
        startedAtRef.current = now
      }
      setElapsedMs(now - startedAtRef.current)
    }, 200)
  }

  function stopTimer() {
    if (timerRef.current) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  function cleanupStream() {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setLiveStream(null)
  }

  function cleanup() {
    stopTimer()
    cleanupStream()
    mediaRecorderRef.current = null
    chunksRef.current = []
  }

  return {
    audioBlob,
    audioUrl,
    elapsedMs,
    error,
    fileName,
    reset,
    start,
    state,
    stop,
    stream: liveStream,
  }
}

function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ]
  return candidates.find((item) => MediaRecorder.isTypeSupported(item)) ?? ""
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function AssessingPanel({ job }: { job: AssessmentJobStatus | null }) {
  const steps = [
    ["upload", "上传录音"],
    ["queued", "等待任务"],
    ["asr_tts", "ASR 与标准音"],
    ["render", "生成声学证据"],
    ["done", "生成报告"],
  ] as const
  const activeStage = job?.stage ?? "upload"
  const activeIndex = Math.max(
    0,
    steps.findIndex(([stage]) => stage === activeStage),
  )

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader className="space-y-4">
          <SectionTitle icon={Activity} title="正在评测" />
          <div className="rounded-panel border border-border bg-background p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xl font-semibold">
                  {job?.message || "录音已提交，正在创建评测任务。"}
                </p>
                <p className="mt-2 text-sm leading-6 text-muted-strong">
                  后端会依次完成 ASR、标准音、F0 提取、MFCC-DTW、五维评分和声学证据生成。
                </p>
              </div>
              <span className="mono rounded-full border border-border bg-panel px-3 py-1 text-sm font-semibold">
                {Math.round(job?.progress ?? 5)}%
              </span>
            </div>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-panel-raised">
              <div
                className="h-full rounded-full bg-foreground transition-all duration-500"
                style={{ width: `${Math.max(5, Math.min(100, job?.progress ?? 5))}%` }}
              />
            </div>
            {job?.error ? (
              <InlineMessage className="mt-4" tone="bad" text={job.error} />
            ) : null}
            {job?.id ? (
              <p className="mono mt-3 text-xs text-muted">任务 ID：{job.id}</p>
            ) : null}
          </div>
        </CardHeader>
      </Card>
      <Card>
        <CardHeader className="space-y-3">
          {steps.map(([stage, label], index) => {
            const isDone = index < activeIndex || job?.status === "done"
            const isActive = index === activeIndex && job?.status !== "done"
            return (
            <div
              className={cn(
                "flex items-center justify-between rounded-panel border px-4 py-3 transition-colors",
                isActive
                  ? "border-foreground bg-panel-raised"
                  : "border-border bg-background",
              )}
              key={stage}
            >
              <span className="text-sm font-medium">{label}</span>
              <span
                className={cn(
                  "flex items-center gap-2 text-xs",
                  isDone ? "text-good" : isActive ? "text-foreground" : "text-muted-strong",
                )}
              >
                {isDone ? (
                  <Check className="size-4" />
                ) : (
                  <Activity className={cn("size-4", isActive && "animate-pulse")} />
                )}
                {isDone ? "完成" : isActive ? "处理中" : "等待"}
              </span>
            </div>
          )})}
        </CardHeader>
      </Card>
    </div>
  )
}

function SyllableTable({ rows }: { rows: SyllableRow[] }) {
  if (!rows.length) {
    return (
      <div className="rounded-panel border border-dashed border-border bg-background p-4 text-sm text-muted-strong">
        本次没有返回逐字诊断。请查看 ASR 和录音质量。
      </div>
    )
  }

  return (
    <div className="min-w-[860px] overflow-hidden rounded-panel border border-border">
      <div className="grid grid-cols-[56px_92px_repeat(7,minmax(80px,1fr))_minmax(160px,1.6fr)] border-b border-border bg-panel-raised px-4 py-3 text-xs font-medium text-muted-strong">
        <span>字</span>
        <span>拼音</span>
        <span>准确度</span>
        <span>声母</span>
        <span>韵母</span>
        <span>发声</span>
        <span>完整度</span>
        <span>声调</span>
        <span>时间</span>
        <span>备注</span>
      </div>
      {rows.map((item, index) => {
        const acc = Number(item.acc_score ?? 100)
        const tone = Number(item.tone_score ?? 100)
        const isWeak =
          Boolean(item.note) || acc < 85 || tone < 80 || item.completeness_ok === false
        return (
          <div
            className={cn(
              "grid grid-cols-[56px_92px_repeat(7,minmax(80px,1fr))_minmax(160px,1.6fr)] items-center border-b border-border/70 px-4 py-3 text-sm last:border-b-0 transition-colors",
              isWeak ? "bg-warn/[0.06] hover:bg-warn/10" : "hover:bg-panel-raised/60",
            )}
            key={`${item.char}-${item.pinyin}-${index}`}
          >
            <span className="text-xl font-semibold">{item.char}</span>
            <span className="mono text-muted-strong">{item.pinyin || "-"}</span>
            <ScoreCell value={item.acc_score} />
            <ScoreCell value={item.initial_score} />
            <ScoreCell value={item.final_score} />
            <ScoreCell value={item.articulation_score} />
            <span>
              <Badge variant={item.completeness_ok === false ? "bad" : "good"}>
                {item.completeness_ok === false ? "疑似漏读" : "已读"}
              </Badge>
            </span>
            <span className="mono">
              {item.expected_tone ?? "?"}→{item.detected_tone ?? "?"}
            </span>
            <span className="mono text-xs text-muted-strong">
              {formatRange(item.start, item.end)}
            </span>
            <span className="truncate text-muted-strong">{item.note || "正常"}</span>
          </div>
        )
      })}
    </div>
  )
}

function AdviceCard({ item }: { item: AdviceItem }) {
  return (
    <div
      className={cn(
        "rounded-panel border px-3 py-2",
        item.level === "good" && "border-good/30 bg-good/8",
        item.level === "warn" && "border-warn/35 bg-warn/10",
        item.level === "bad" && "border-bad/35 bg-bad/10",
      )}
    >
      <div className="flex items-start gap-2">
        <span
          className={cn(
            "mt-1 size-2 rounded-full",
            item.level === "good" && "bg-good",
            item.level === "warn" && "bg-warn",
            item.level === "bad" && "bg-bad",
          )}
        />
        <div className="min-w-0">
          <p className="text-sm font-semibold">{item.title}</p>
          <p className="mt-1 text-sm leading-5 text-muted-strong">{item.detail}</p>
        </div>
      </div>
    </div>
  )
}

function CompareBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-panel border border-border bg-background p-3">
      <p className="mb-1 text-xs text-muted">{label}</p>
      <p className="text-sm leading-6 text-foreground">{value}</p>
    </div>
  )
}

function ScoreCell({ value }: { value?: number | null }) {
  if (value === null || value === undefined) {
    return <span className="text-muted">-</span>
  }
  return <span className={cn("mono", scoreText(value))}>{value.toFixed(1)}</span>
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

function InlineMessage({
  className,
  text,
  tone,
}: {
  className?: string
  text: string
  tone: "bad" | "warn"
}) {
  return (
    <p
      className={cn(
        "rounded-panel border px-3 py-2 text-xs leading-5",
        tone === "bad"
          ? "border-bad/35 bg-bad/10 text-bad"
          : "border-warn/35 bg-warn/10 text-warn",
        className,
      )}
    >
      {text}
    </p>
  )
}

function scoreTone(value: number): "good" | "warn" | "bad" | "neutral" {
  if (value >= 85) {
    return "good"
  }
  if (value >= 70) {
    return "warn"
  }
  return "bad"
}

function scoreText(value: number) {
  if (value >= 90) {
    return "text-good"
  }
  if (value >= 70) {
    return "text-warn"
  }
  return "text-bad"
}

function formatElapsed(ms: number) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes.toString().padStart(2, "0")}:${seconds
    .toString()
    .padStart(2, "0")}`
}

function formatRange(start?: number, end?: number) {
  if (start === undefined || end === undefined) {
    return "--"
  }
  return `${start.toFixed(2)}-${end.toFixed(2)}`
}

export default App
