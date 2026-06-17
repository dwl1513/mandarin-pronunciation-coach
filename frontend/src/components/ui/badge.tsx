import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "../../lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium leading-none",
  {
    variants: {
      variant: {
        default: "border-border-strong bg-foreground text-background",
        outline: "border-border text-muted-strong",
        good: "border-good/40 bg-good/10 text-good",
        warn: "border-warn/40 bg-warn/10 text-warn",
        bad: "border-bad/45 bg-bad/12 text-bad",
      },
    },
    defaultVariants: {
      variant: "outline",
    },
  },
)

interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge }
