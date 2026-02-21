'use client';

import type { SimulationComponentProps } from '@/shared/types/simulation';

export function GaussianElimDemo({}: SimulationComponentProps) {
  return (
    <div className="flex items-center justify-center rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]/30 p-8">
      <p className="text-sm text-[var(--text-muted)]">
        Gaussian elimination visualization — coming soon.
      </p>
    </div>
  );
}
