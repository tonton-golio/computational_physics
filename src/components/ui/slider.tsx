import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"

import { cn } from "@/lib/utils"

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "group relative flex w-full touch-none select-none items-center py-1",
      className
    )}
    {...props}
  >
    <SliderPrimitive.Track className="relative h-2.5 w-full grow overflow-hidden rounded-full border border-[var(--border-strong)] bg-[var(--surface-2)]/95 shadow-inner">
      <SliderPrimitive.Range className="absolute h-full rounded-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-strong)]" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-[var(--accent)] bg-[var(--surface-1)] shadow-[0_1px_3px_rgba(16,26,43,0.3)] transition-[transform,box-shadow,border-color] duration-150 hover:scale-110 hover:shadow-[0_0_0_4px_var(--accent)]/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]/45 focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--background)] active:scale-95 group-data-[disabled]:opacity-50 disabled:pointer-events-none" />
  </SliderPrimitive.Root>
))
Slider.displayName = SliderPrimitive.Root.displayName

export { Slider }
