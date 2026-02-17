import * as React from "react";

import { cn } from "@/lib/utils";

const Label = React.forwardRef<
  React.ElementRef<"label">,
  React.ComponentPropsWithoutRef<"label">
>(({ className, ...props }, ref) => (
  <label
    ref={ref}
    className={cn(
      "text-sm font-medium leading-none text-[var(--text-strong)]",
      className
    )}
    {...props}
  />
));

Label.displayName = "Label";

export { Label };
