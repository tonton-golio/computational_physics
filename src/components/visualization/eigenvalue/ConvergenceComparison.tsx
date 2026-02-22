"use client";

import { useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { COLORS } from '@/lib/chart-colors';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function ConvergenceComparison({}: SimulationComponentProps) {
  const { data, layout } = useMemo(() => {
    const iterations = 20;
    const trueEigenvalue = 5;

    // Power method: linear convergence
    const powerMethod = Array.from({ length: iterations }, (_, i) =>
      trueEigenvalue - 2 * Math.exp(-0.3 * i)
    );

    // Inverse iteration: linear but faster
    const inverseIter = Array.from({ length: iterations }, (_, i) =>
      trueEigenvalue - 1.5 * Math.exp(-0.5 * i)
    );

    // Rayleigh quotient: cubic convergence
    const rayleigh = Array.from({ length: iterations }, (_, i) =>
      i < 5 ? trueEigenvalue - Math.exp(-0.8 * i * i) : trueEigenvalue
    );

    const x = Array.from({ length: iterations }, (_, i) => i);

    const data = [
      {
        x,
        y: powerMethod.map(v => Math.abs(v - trueEigenvalue)),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        line: { color: COLORS.primary, width: 2 },
        marker: { size: 6 },
        name: 'Power Method (linear)',
      },
      {
        x,
        y: inverseIter.map(v => Math.abs(v - trueEigenvalue)),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        line: { color: COLORS.secondary, width: 2 },
        marker: { size: 6 },
        name: 'Inverse Iteration (linear)',
      },
      {
        x,
        y: rayleigh.map(v => Math.abs(v - trueEigenvalue) + 1e-15),
        type: 'scatter' as const,
        mode: 'lines+markers' as const,
        line: { color: COLORS.tertiary, width: 2 },
        marker: { size: 6 },
        name: 'Rayleigh Quotient (cubic)',
      },
    ];

    const layout = {
      xaxis: { title: { text: 'Iteration' } },
      yaxis: { title: { text: 'Error |\u03BB - \u03BB_true|' }, type: 'log' as const },
      title: { text: 'Eigenvalue Algorithm Convergence' },
    };

    return { data, layout };
  }, []);

  return (
    <SimulationPanel title="Eigenvalue Algorithm Convergence" caption="Rayleigh quotient iteration achieves cubic convergence, dramatically faster than the linear convergence of power method.">
      <SimulationMain>
        <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 320 }} />
      </SimulationMain>
    </SimulationPanel>
  );
}
