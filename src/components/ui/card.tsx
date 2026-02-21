"use client";

import { useSimulationFullscreen } from "@/lib/simulation-fullscreen-context";

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export function Card({ children, className = "" }: CardProps) {
  const isFullscreen = useSimulationFullscreen();
  return (
    <div
      data-fs-role={isFullscreen ? "controls" : undefined}
      className={`rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] p-6 shadow-sm ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className = "" }: CardProps) {
  return <div className={`mb-4 ${className}`}>{children}</div>;
}

export function CardTitle({ children, className = "" }: CardProps) {
  return <h3 className={`text-xl font-semibold text-[var(--text-strong)] ${className}`}>{children}</h3>;
}

export function CardContent({ children, className = "" }: CardProps) {
  return <div className={className}>{children}</div>;
}
