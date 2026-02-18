import * as React from "react"
import { cn } from "@/lib/utils"

interface SimulationPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string
  description?: string
  children: React.ReactNode
}

function SimulationPanel({
  title,
  description,
  children,
  className,
  ...props
}: SimulationPanelProps) {
  return (
    <div
      className={cn(
        "w-full rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/80 backdrop-blur-sm p-6 mb-8 shadow-[0_1px_3px_rgba(0,0,0,0.06),0_1px_2px_rgba(0,0,0,0.04)] overflow-hidden",
        className,
      )}
      {...props}
    >
      {/* Accent gradient bar */}
      <div
        className="h-[2px] -mx-6 -mt-6 mb-5"
        style={{ background: "linear-gradient(90deg, var(--accent), var(--accent-strong), var(--accent))" }}
      />
      {title && (
        <h3 className="text-lg font-semibold mb-4 text-[var(--text-strong)] font-mono tracking-tight">
          {title}
        </h3>
      )}
      {description && (
        <p className="text-[var(--text-muted)] text-sm mb-4">{description}</p>
      )}
      {children}
    </div>
  )
}

function SimulationLabel({
  className,
  children,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return (
    <label
      className={cn(
        "block text-sm font-medium text-[var(--text-muted)] mb-1.5",
        className,
      )}
      {...props}
    >
      {children}
    </label>
  )
}

interface SimulationToggleProps {
  options: { label: string; value: string }[]
  value: string
  onChange: (value: string) => void
  className?: string
}

function SimulationToggle({
  options,
  value,
  onChange,
  className,
}: SimulationToggleProps) {
  return (
    <div
      className={cn(
        "inline-flex rounded-lg border border-[var(--border-strong)] bg-[var(--surface-2)] p-0.5",
        className,
      )}
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            "px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
            value === opt.value
              ? "bg-[var(--accent)] text-white shadow-sm"
              : "text-[var(--text-muted)] hover:text-[var(--text-strong)]",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

function SimulationControls({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-end gap-6 py-3",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export { SimulationPanel, SimulationLabel, SimulationToggle, SimulationControls }
