"use client";

import * as React from "react"
import { cn } from "@/lib/utils"
import { useSimulationFullscreen } from "@/lib/simulation-fullscreen-context"
import { SimulationMainContext } from "@/components/ui/simulation-main"

interface SimulationPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string
  caption?: string
  children?: React.ReactNode
}

function SimulationPanel({
  title,
  caption,
  children,
  className,
  ...props
}: SimulationPanelProps) {
  const isFullscreen = useSimulationFullscreen();
  const captionText = caption;
  return (
    <div
      className={cn(
        "w-full rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/80 backdrop-blur-sm p-6 mb-8 shadow-[0_1px_3px_rgba(0,0,0,0.06),0_1px_2px_rgba(0,0,0,0.04)] overflow-hidden",
        className,
      )}
      {...props}
    >
      {/* Accent gradient bar — hidden in fullscreen (SimulationHost provides its own) */}
      {!isFullscreen && (
        <div
          className="h-[2px] -mx-6 -mt-6 mb-5"
          style={{ background: "linear-gradient(90deg, var(--accent), var(--accent-strong), var(--accent))" }}
        />
      )}
      {title && !isFullscreen && (
        <h3 className="text-lg font-semibold mb-4 text-[var(--text-strong)] font-mono tracking-tight">
          {title}
        </h3>
      )}
      {title && isFullscreen && (
        <div data-sim-slot="title" className="sim-fs-title-center">{title}</div>
      )}
      {captionText && !isFullscreen && (
        <p className="text-[var(--text-muted)] text-sm mb-4">{captionText}</p>
      )}
      {captionText && isFullscreen && (
        <div data-sim-slot="caption" className="text-sm text-[var(--text-muted)]">{captionText}</div>
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

/* ---------- SimulationSettings ---------- */

function SimulationSettings({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      data-sim-slot="settings"
      className={cn("flex flex-wrap items-center gap-3 py-3", className)}
      {...props}
    >
      {children}
    </div>
  );
}

/* ---------- SimulationConfig ---------- */

function SimulationConfig({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  const isFullscreen = useSimulationFullscreen();
  return (
    <div
      data-sim-slot="config"
      className={cn(
        !isFullscreen && "flex flex-wrap items-end gap-6 py-3",
        isFullscreen && "flex flex-col gap-3",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

/* ---------- SimulationResults ---------- */

interface SimulationResultsProps extends React.HTMLAttributes<HTMLDivElement> {
  alert?: React.ReactNode;
}

function SimulationResults({
  alert: alertContent,
  className,
  children,
  ...props
}: SimulationResultsProps) {
  return (
    <div data-sim-slot="results" className={cn("space-y-2 py-2", className)} {...props}>
      {alertContent && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
          {alertContent}
        </div>
      )}
      {children}
    </div>
  );
}

/* ---------- SimulationAux ---------- */

function SimulationAux({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <SimulationMainContext.Provider value="aux">
      <div data-sim-slot="aux" className={cn("space-y-4", className)} {...props}>
        {children}
      </div>
    </SimulationMainContext.Provider>
  );
}

/* ---------- SimulationButton ---------- */

const BUTTON_VARIANTS = {
  primary:
    "bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white",
  secondary:
    "bg-[var(--surface-3)] hover:bg-[var(--border-strong)] text-[var(--text-strong)]",
  danger:
    "bg-red-600 hover:bg-red-700 text-white",
} as const

interface SimulationButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof BUTTON_VARIANTS
}

function SimulationButton({
  variant = "secondary",
  className,
  ...props
}: SimulationButtonProps) {
  return (
    <button
      className={cn(
        "px-4 py-1.5 rounded text-sm font-medium transition-colors disabled:opacity-40",
        BUTTON_VARIANTS[variant],
        className,
      )}
      {...props}
    />
  )
}

/* ---------- SimulationPlayButton ---------- */

interface SimulationPlayButtonProps {
  isRunning: boolean
  onToggle: () => void
  className?: string
  disabled?: boolean
  labels?: { play?: string; pause?: string }
}

function SimulationPlayButton({
  isRunning,
  onToggle,
  className,
  disabled,
  labels,
}: SimulationPlayButtonProps) {
  const playLabel = labels?.play ?? "Play"
  const pauseLabel = labels?.pause ?? "Pause"
  return (
    <SimulationButton
      variant={isRunning ? "danger" : "primary"}
      onClick={onToggle}
      className={className}
      disabled={disabled}
    >
      {isRunning ? pauseLabel : playLabel}
    </SimulationButton>
  )
}

/* ---------- SimulationCheckbox ---------- */

interface SimulationCheckboxProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  className?: string
  style?: React.CSSProperties
}

function SimulationCheckbox({
  checked,
  onChange,
  label,
  className,
  style,
}: SimulationCheckboxProps) {
  return (
    <label
      className={cn(
        "flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer select-none",
        className,
      )}
      style={style}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-[var(--accent)]"
      />
      {label}
    </label>
  )
}

export {
  SimulationPanel,
  SimulationLabel,
  SimulationToggle,
  SimulationSettings,
  SimulationConfig,
  SimulationResults,
  SimulationAux,
  SimulationButton,
  SimulationPlayButton,
  SimulationCheckbox,
}
