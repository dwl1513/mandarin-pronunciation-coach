import { cn } from "../../lib/utils"

type MeterProps = {
  value: number
  label?: string
  tone?: "neutral" | "good" | "warn" | "bad"
  className?: string
}

const toneClass = {
  neutral: "bg-foreground",
  good: "bg-good",
  warn: "bg-warn",
  bad: "bg-bad",
}

export function Meter({ value, label, tone = "neutral", className }: MeterProps) {
  const width = Math.max(0, Math.min(100, value))
  return (
    <div className={cn("space-y-2", className)}>
      {label ? (
        <div className="flex items-center justify-between text-xs text-muted-strong">
          <span>{label}</span>
          <span className="mono">{value.toFixed(1)}</span>
        </div>
      ) : null}
      <div className="h-2 overflow-hidden rounded-full bg-foreground/10">
        <div
          className={cn("h-full rounded-full transition-all duration-500", toneClass[tone])}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  )
}
