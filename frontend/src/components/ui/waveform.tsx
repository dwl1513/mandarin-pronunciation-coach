import { useEffect, useRef } from "react"

import { cn } from "../../lib/utils"

type LiveWaveformProps = {
  /** 录音中的实时音频流，用于绘制滚动波形 */
  stream: MediaStream | null
  /** 录制完成后的音频 Blob，用于解码并绘制整段静态波形 */
  blob: Blob | null
  state: "idle" | "recording" | "recorded"
  className?: string
}

const WAVE_COLOR = "oklch(0.17 0 0)"
const MUTED_COLOR = "oklch(0.72 0 0)"
const MID_LINE = "oklch(0.895 0 0)"

/**
 * 真实录音波形：
 * - recording 态：用 AnalyserNode 取时域数据，画从右向左滚动的实时波形
 * - recorded 态：用 decodeAudioData 解码整段录音，按峰值降采样画完整波形
 * - idle 态：画一条静音中线
 */
export function LiveWaveform({ stream, blob, state, className }: LiveWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const historyRef = useRef<number[]>([])

  // 自适应画布像素密度，避免在高分屏上糊
  function setupCanvas(canvas: HTMLCanvasElement) {
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = Math.max(1, Math.floor(rect.width * dpr))
    canvas.height = Math.max(1, Math.floor(rect.height * dpr))
    const ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    return { ctx, width: rect.width, height: rect.height }
  }

  function drawMidline(ctx: CanvasRenderingContext2D, width: number, height: number) {
    ctx.clearRect(0, 0, width, height)
    ctx.strokeStyle = MID_LINE
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, height / 2)
    ctx.lineTo(width, height / 2)
    ctx.stroke()
  }

  // ----- 实时滚动波形 -----
  useEffect(() => {
    if (state !== "recording" || !stream) {
      return
    }
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    const setup = setupCanvas(canvas)
    const ctx = setup.ctx
    if (!ctx) {
      return
    }
    const { width, height } = setup
    historyRef.current = []

    const AudioCtor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioCtor) {
      return
    }
    const audioCtx = new AudioCtor()
    audioCtxRef.current = audioCtx
    const analyser = audioCtx.createAnalyser()
    analyser.fftSize = 1024
    const source = audioCtx.createMediaStreamSource(stream)
    source.connect(analyser)
    const data = new Uint8Array(analyser.frequencyBinCount)
    const maxBars = Math.floor(width / 3)

    const tick = () => {
      analyser.getByteTimeDomainData(data)
      let peak = 0
      for (const v of data) {
        const centered = Math.abs(v - 128) / 128
        if (centered > peak) {
          peak = centered
        }
      }
      const history = historyRef.current
      history.push(peak)
      if (history.length > maxBars) {
        history.shift()
      }

      ctx.clearRect(0, 0, width, height)
      ctx.strokeStyle = MID_LINE
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, height / 2)
      ctx.lineTo(width, height / 2)
      ctx.stroke()

      ctx.fillStyle = WAVE_COLOR
      const barW = 2
      const gap = 1
      history.forEach((amp, i) => {
        const x = width - (history.length - i) * (barW + gap)
        const barH = Math.max(2, amp * height * 0.92)
        ctx.fillRect(x, (height - barH) / 2, barW, barH)
      })

      rafRef.current = window.requestAnimationFrame(tick)
    }
    tick()

    return () => {
      if (rafRef.current) {
        window.cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      source.disconnect()
      void audioCtx.close()
      audioCtxRef.current = null
    }
  }, [state, stream])

  // ----- 录制完成后的整段静态波形 -----
  useEffect(() => {
    if (state !== "recorded" || !blob) {
      return
    }
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    let cancelled = false

    const render = async () => {
      const setup = setupCanvas(canvas)
      const ctx = setup.ctx
      if (!ctx) {
        return
      }
      const { width, height } = setup
      drawMidline(ctx, width, height)

      try {
        const buffer = await blob.arrayBuffer()
        const AudioCtor =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
        if (!AudioCtor) {
          return
        }
        const audioCtx = new AudioCtor()
        const decoded = await audioCtx.decodeAudioData(buffer)
        void audioCtx.close()
        if (cancelled) {
          return
        }

        const channel = decoded.getChannelData(0)
        const barW = 2
        const gap = 1
        const bars = Math.floor(width / (barW + gap))
        const blockSize = Math.floor(channel.length / bars) || 1

        ctx.clearRect(0, 0, width, height)
        ctx.strokeStyle = MID_LINE
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(0, height / 2)
        ctx.lineTo(width, height / 2)
        ctx.stroke()

        ctx.fillStyle = WAVE_COLOR
        for (let i = 0; i < bars; i++) {
          let peak = 0
          const start = i * blockSize
          for (let j = 0; j < blockSize; j++) {
            const sample = Math.abs(channel[start + j] || 0)
            if (sample > peak) {
              peak = sample
            }
          }
          const barH = Math.max(2, peak * height * 0.92)
          ctx.fillRect(i * (barW + gap), (height - barH) / 2, barW, barH)
        }
      } catch {
        // 解码失败（某些 webm 在部分浏览器无法解码）时退化为一条中线
        drawMidline(ctx, width, height)
        ctx.fillStyle = MUTED_COLOR
        ctx.font = "12px sans-serif"
        ctx.textAlign = "center"
        ctx.fillText("录音已就绪（波形预览不可用）", width / 2, height / 2 - 8)
      }
    }
    void render()

    return () => {
      cancelled = true
    }
  }, [state, blob])

  // idle 态画中线
  useEffect(() => {
    if (state !== "idle") {
      return
    }
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }
    const setup = setupCanvas(canvas)
    if (setup.ctx) {
      drawMidline(setup.ctx, setup.width, setup.height)
    }
  }, [state])

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-panel border border-border bg-background",
        className,
      )}
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      <div className="absolute right-3 top-3 flex items-center gap-1.5 rounded-full border border-border bg-panel/90 px-2 py-1 text-xs text-muted-strong backdrop-blur">
        <span
          className={cn(
            "size-2 rounded-full",
            state === "recording" ? "animate-pulse bg-bad" : "bg-muted",
          )}
        />
        {state === "recording" ? "录音中" : state === "recorded" ? "已录音" : "待录音"}
      </div>
    </div>
  )
}
