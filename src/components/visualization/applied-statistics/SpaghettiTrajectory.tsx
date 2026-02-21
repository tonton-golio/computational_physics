'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

export default function SpaghettiTrajectory() {
  const [nTrajectories, setNTrajectories] = useState(20);
  const [drift, setDrift] = useState(0.5);

  const { tVals, trajectories, meanTraj, upperBand, lowerBand } = useMemo(() => {
    const nSteps = 50;
    const dt = 1;
    const ts: number[] = [];
    for (let i = 0; i <= nSteps; i++) ts.push(i * dt);

    const trajs: number[][] = [];
    for (let t = 0; t < nTrajectories; t++) {
      const path: number[] = [0];
      for (let i = 1; i <= nSteps; i++) {
        const noise = Math.sin((t + 1) * 1.37 * i + t * 7.13) * 2;
        path.push(path[i - 1] + drift * dt + noise * Math.sqrt(dt));
      }
      trajs.push(path);
    }

    const mean: number[] = [];
    const upper: number[] = [];
    const lower: number[] = [];
    for (let i = 0; i <= nSteps; i++) {
      const vals = trajs.map((tr) => tr[i]);
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      const std = Math.sqrt(vals.reduce((a, v) => a + (v - m) ** 2, 0) / vals.length);
      mean.push(m);
      upper.push(m + 2 * std);
      lower.push(m - 2 * std);
    }

    return { tVals: ts, trajectories: trajs, meanTraj: mean, upperBand: upper, lowerBand: lower };
  }, [nTrajectories, drift]);

  const pathTraces: any[] = trajectories.map((tr, i) => ({
    x: tVals, y: tr, type: 'scatter', mode: 'lines',
    line: { color: '#3b82f6', width: 0.8 },
    opacity: 0.3,
    showlegend: i === 0,
    name: i === 0 ? 'Individual paths' : undefined,
  }));

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Spaghetti Plot: Random Trajectories</h3>
      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Trajectories: {nTrajectories}</label>
          <Slider value={[nTrajectories]} onValueChange={([v]) => setNTrajectories(v)} min={5} max={50} step={1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Drift: {drift.toFixed(2)}</label>
          <Slider value={[drift]} onValueChange={([v]) => setDrift(v)} min={-1} max={2} step={0.05} />
        </div>
      </div>
      <CanvasChart
        data={[
          { x: tVals, y: upperBand, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 0 }, showlegend: false },
          { x: tVals, y: lowerBand, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 0 }, fill: 'tonexty', fillcolor: 'rgba(239,68,68,0.12)', name: '95% band' },
          ...pathTraces,
          { x: tVals, y: meanTraj, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 3 }, name: 'Mean trajectory' },
        ]}
        layout={{
          height: 420,
          xaxis: { title: { text: 'Time step' } },
          yaxis: { title: { text: 'Value' } },
        }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
