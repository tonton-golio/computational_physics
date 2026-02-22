"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Demonstrates how density fluctuations decrease as the averaging volume grows.
 * A random field of particles is binned at varying resolutions; the relative
 * fluctuation (std / mean) is plotted against the number of particles per cell,
 * showing the 1/sqrt(N) convergence that underpins the continuum approximation.
 */

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

export default function AveragingVolume({}: SimulationComponentProps) {
  const [nParticles, setNParticles] = useState(2000);
  const [seed] = useState(42);

  const { fluctData, theoreticalData } = useMemo(() => {
    const rng = seededRandom(seed);
    const particles = Array.from({ length: nParticles }, () => ({
      x: rng(),
      y: rng(),
    }));

    const gridSizes = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50];
    const avgNArr: number[] = [];
    const fluctArr: number[] = [];

    for (const g of gridSizes) {
      const cellSize = 1 / g;
      const nCells = g * g;
      const counts = new Array(nCells).fill(0);
      for (const p of particles) {
        const col = Math.min(g - 1, Math.floor(p.x / cellSize));
        const row = Math.min(g - 1, Math.floor(p.y / cellSize));
        counts[row * g + col]++;
      }
      const mean = counts.reduce((a, b) => a + b, 0) / nCells;
      if (mean < 0.5) continue;
      const variance = counts.reduce((a, c) => a + (c - mean) ** 2, 0) / nCells;
      const relFluct = Math.sqrt(variance) / mean;
      avgNArr.push(mean);
      fluctArr.push(relFluct);
    }

    // Theoretical 1/sqrt(N) curve
    const theoN: number[] = [];
    const theoF: number[] = [];
    const nMin = Math.max(0.5, Math.min(...avgNArr));
    const nMax = Math.max(...avgNArr);
    for (let i = 0; i <= 100; i++) {
      const n = nMin * Math.pow(nMax / nMin, i / 100);
      theoN.push(n);
      theoF.push(1 / Math.sqrt(n));
    }

    return {
      fluctData: { x: avgNArr, y: fluctArr },
      theoreticalData: { x: theoN, y: theoF },
    };
  }, [nParticles, seed]);

  const data = useMemo(() => [
    {
      type: 'scatter' as const,
      mode: 'markers' as const,
      x: fluctData.x,
      y: fluctData.y,
      name: 'Measured fluctuation',
      marker: { color: '#3b82f6', size: 7 },
    },
    {
      type: 'scatter' as const,
      mode: 'lines' as const,
      x: theoreticalData.x,
      y: theoreticalData.y,
      name: '1/\u221AN (theory)',
      line: { color: '#ef4444', width: 2, dash: 'dash' as const },
    },
  ], [fluctData, theoreticalData]);

  const layout = useMemo(() => ({
    xaxis: { title: { text: 'Avg. particles per cell (N)' }, type: 'log' as const },
    yaxis: { title: { text: 'Relative fluctuation \u0394\u03c1/\u03c1' }, type: 'log' as const },
  }), []);

  return (
    <SimulationPanel title="Averaging Volume and Density Convergence" caption="Random particles are binned at different grid resolutions. As each cell contains more particles, the relative density fluctuation drops as 1/\u221AN \u2014 the continuum approximation becomes valid.">
      <SimulationConfig>
        <div className="max-w-xs">
          <SimulationLabel>
            Total particles: {nParticles}
          </SimulationLabel>
          <Slider
            min={200}
            max={5000}
            step={100}
            value={[nParticles]}
            onValueChange={([v]) => setNParticles(v)}
            className="w-full"
          />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 420 }} />
      </SimulationMain>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Blue markers: measured std/mean at each grid resolution.
        Red dashed: theoretical 1/&radic;N scaling.
        Increase the particle count to push the continuum regime to finer grids.
      </p>
    </SimulationPanel>
  );
}
