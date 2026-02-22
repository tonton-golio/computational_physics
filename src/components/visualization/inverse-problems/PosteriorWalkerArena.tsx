"use client";

import { useState, useMemo } from 'react';
import { mulberry32, gaussianPair } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

// 2D bimodal posterior: sum of two Gaussians with correlation
function logPosterior(x: number, y: number): number {
  const dx1 = x - 1.5, dy1 = y - 1.5;
  const dx2 = x + 1.5, dy2 = y + 1.5;
  const rho = 0.6;
  const s1 = 0.8, s2 = 0.8;
  const z1 = (dx1 * dx1 / (s1 * s1) - 2 * rho * dx1 * dy1 / (s1 * s2) + dy1 * dy1 / (s2 * s2)) / (2 * (1 - rho * rho));
  const z2 = (dx2 * dx2 / (s1 * s1) - 2 * rho * dx2 * dy2 / (s1 * s2) + dy2 * dy2 / (s2 * s2)) / (2 * (1 - rho * rho));
  const p1 = Math.exp(-z1);
  const p2 = 0.7 * Math.exp(-z2);
  return Math.log(p1 + p2 + 1e-30);
}

export default function PosteriorWalkerArena({}: SimulationComponentProps) {
  const [nWalkers, setNWalkers] = useState(4);
  const [stepSize, setStepSize] = useState(0.5);
  const [nSteps, setNSteps] = useState(200);
  const [burnIn, setBurnIn] = useState(50);

  const result = useMemo(() => {
    const rng = mulberry32(7);
    const walkerTraces: { x: number[]; y: number[] }[] = [];
    const burnSamples: { x: number[]; y: number[] } = { x: [], y: [] };
    const postSamples: { x: number[]; y: number[] } = { x: [], y: [] };

    for (let w = 0; w < nWalkers; w++) {
      let cx = (rng() - 0.5) * 6;
      let cy = (rng() - 0.5) * 6;
      let curLP = logPosterior(cx, cy);
      const trX: number[] = [cx];
      const trY: number[] = [cy];

      for (let s = 0; s < nSteps; s++) {
        const px = cx + gaussianPair(rng)[0] * stepSize;
        const py = cy + gaussianPair(rng)[0] * stepSize;
        const propLP = logPosterior(px, py);

        if (Math.log(rng()) < propLP - curLP) {
          cx = px; cy = py; curLP = propLP;
        }
        trX.push(cx);
        trY.push(cy);

        if (s < burnIn) {
          burnSamples.x.push(cx);
          burnSamples.y.push(cy);
        } else {
          postSamples.x.push(cx);
          postSamples.y.push(cy);
        }
      }
      walkerTraces.push({ x: trX, y: trY });
    }

    // Generate contour lines of the posterior as iso-level curves
    const gridSize = 50;
    const lo = -4.5, hi = 4.5;
    const step = (hi - lo) / (gridSize - 1);
    const contourLines: { x: number[]; y: number[] }[] = [];
    const levels = [-0.5, -1.0, -2.0, -3.5, -5.0];

    for (const level of levels) {
      const pts: { x: number; y: number }[] = [];
      for (let i = 0; i < gridSize - 1; i++) {
        for (let j = 0; j < gridSize - 1; j++) {
          const x0 = lo + j * step, y0 = lo + i * step;
          const x1 = x0 + step, y1 = y0 + step;
          const v00 = logPosterior(x0, y0) - level;
          const v10 = logPosterior(x1, y0) - level;
          const v01 = logPosterior(x0, y1) - level;
          const v11 = logPosterior(x1, y1) - level;

          // Marching squares: find zero crossings on edges
          const edges: [number, number][] = [];
          if (v00 * v10 < 0) edges.push([x0 + step * (-v00) / (v10 - v00), y0]);
          if (v10 * v11 < 0) edges.push([x1, y0 + step * (-v10) / (v11 - v10)]);
          if (v01 * v11 < 0) edges.push([x0 + step * (-v01) / (v11 - v01), y1]);
          if (v00 * v01 < 0) edges.push([x0, y0 + step * (-v00) / (v01 - v00)]);

          for (const [ex, ey] of edges) pts.push({ x: ex, y: ey });
        }
      }
      if (pts.length > 0) {
        contourLines.push({ x: pts.map(p => p.x), y: pts.map(p => p.y) });
      }
    }

    return { walkerTraces, burnSamples, postSamples, contourLines };
  }, [nWalkers, stepSize, nSteps, burnIn]);

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <SimulationPanel title="Posterior Walker Arena">
      <SimulationConfig>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <SimulationLabel className="mb-1 block text-sm text-[var(--text-muted)]">Walkers: {nWalkers}</SimulationLabel>
            <Slider value={[nWalkers]} onValueChange={([v]) => setNWalkers(v)} min={1} max={8} step={1} />
          </div>
          <div>
            <SimulationLabel className="mb-1 block text-sm text-[var(--text-muted)]">Step size: {stepSize.toFixed(2)}</SimulationLabel>
            <Slider value={[stepSize]} onValueChange={([v]) => setStepSize(v)} min={0.05} max={2.0} step={0.05} />
          </div>
          <div>
            <SimulationLabel className="mb-1 block text-sm text-[var(--text-muted)]">Steps: {nSteps}</SimulationLabel>
            <Slider value={[nSteps]} onValueChange={([v]) => setNSteps(v)} min={50} max={500} step={10} />
          </div>
          <div>
            <SimulationLabel className="mb-1 block text-sm text-[var(--text-muted)]">Burn-in: {burnIn}</SimulationLabel>
            <Slider value={[burnIn]} onValueChange={([v]) => setBurnIn(v)} min={0} max={200} step={10} />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <CanvasChart
          data={[
            // Contour lines
            ...result.contourLines.map((cl, i) => ({
              x: cl.x, y: cl.y,
              type: 'scatter' as const, mode: 'markers' as const,
              marker: { size: 1.5, color: `rgba(100,200,100,${0.15 + i * 0.1})` },
              showlegend: i === 0,
              name: i === 0 ? 'Posterior contours' : '',
            })),
            // Walker trajectories
            ...result.walkerTraces.map((tr, w) => ({
              x: tr.x, y: tr.y,
              type: 'scatter' as const, mode: 'lines' as const,
              line: { color: COLORS[w % COLORS.length], width: 1 },
              opacity: 0.7,
              name: `Walker ${w + 1}`,
            })),
          ]}
          layout={{
            title: { text: 'Walker Trajectories on Posterior' },
            xaxis: { title: { text: 'x' }, range: [-4.5, 4.5] },
            yaxis: { title: { text: 'y' }, range: [-4.5, 4.5] },
            height: 400,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />

        <CanvasChart
          data={[
            {
              x: result.burnSamples.x, y: result.burnSamples.y,
              type: 'scatter', mode: 'markers',
              marker: { size: 3, color: 'rgba(239,68,68,0.4)' },
              name: 'Burn-in samples',
            },
            {
              x: result.postSamples.x, y: result.postSamples.y,
              type: 'scatter', mode: 'markers',
              marker: { size: 3, color: 'rgba(59,130,246,0.5)' },
              name: 'Post-burn-in samples',
            },
          ]}
          layout={{
            title: { text: 'Sample Distribution' },
            xaxis: { title: { text: 'x' }, range: [-4.5, 4.5] },
            yaxis: { title: { text: 'y' }, range: [-4.5, 4.5] },
            height: 400,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          The posterior is bimodal (two peaks). Walkers must find both modes to characterize the
          full distribution. Small step sizes keep walkers stuck in one mode; large steps cause
          high rejection rates. The burn-in phase (red) shows initial exploration before walkers
          settle into the high-probability regions (blue).
        </p>
      </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
