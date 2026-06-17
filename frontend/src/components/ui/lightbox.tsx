import { useEffect } from "react"
import { X, ZoomIn } from "lucide-react"

import { cn } from "../../lib/utils"

type LightboxProps = {
  src: string
  alt: string
  caption?: string
  onClose: () => void
}

/** 全屏看图：点击声学证据图后放大查看细节，ESC 或点遮罩关闭 */
export function Lightbox({ src, alt, caption, onClose }: LightboxProps) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose()
      }
    }
    window.addEventListener("keydown", onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = "hidden"
    return () => {
      window.removeEventListener("keydown", onKey)
      document.body.style.overflow = prev
    }
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-foreground/80 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={alt}
    >
      <div className="flex shrink-0 items-center justify-between px-5 py-4 text-background">
        <p className="text-sm font-medium">{alt}</p>
        <button
          aria-label="关闭"
          className="flex size-9 items-center justify-center rounded-full bg-background/15 text-background transition hover:bg-background/25"
          onClick={onClose}
          type="button"
        >
          <X className="size-5" />
        </button>
      </div>
      <button
        aria-label="关闭"
        className="flex min-h-0 flex-1 cursor-zoom-out items-center justify-center px-5 pb-5"
        onClick={onClose}
        type="button"
      >
        <img
          alt={alt}
          className="max-h-full max-w-full rounded-panel bg-white object-contain shadow-2xl"
          onClick={(event) => event.stopPropagation()}
          src={src}
        />
      </button>
      {caption ? (
        <div className="shrink-0 px-5 pb-5 text-center text-sm leading-6 text-background/80">
          {caption}
        </div>
      ) : null}
    </div>
  )
}

/** 可点击放大的证据图包装，悬停显示放大提示 */
export function ZoomableImage({
  src,
  alt,
  className,
  onZoom,
}: {
  src: string
  alt: string
  className?: string
  onZoom: () => void
}) {
  return (
    <button
      className={cn(
        "group relative block w-full cursor-zoom-in overflow-hidden rounded-panel border border-border bg-white",
        className,
      )}
      onClick={onZoom}
      type="button"
      aria-label={`放大查看${alt}`}
    >
      <img alt={alt} className="block w-full" src={src} />
      <span className="absolute right-3 top-3 flex items-center gap-1.5 rounded-full bg-foreground/75 px-2.5 py-1 text-xs font-medium text-background opacity-0 transition group-hover:opacity-100">
        <ZoomIn className="size-3.5" />
        点击放大
      </span>
    </button>
  )
}
