"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function ValidationSplitDemo({}: SimulationComponentProps): React.ReactElement {
  const [epochs, setEpochs] = useState(50);
  const [trainPct, setTrainPct] = useState(60);
  const [valPct, setValPct] = useState(20);
  const testPct = 100 - trainPct - valPct;
  const overfitStart = useMemo(() => Math.max(8, Math.round(35 * (trainPct / 80))), [trainPct]);

  // Less training data = noisier curves and earlier overfitting
  const noiseFactor = useMemo(() => Math.max(0.5, (80 - trainPct) / 40), [trainPct]);

  const curves = useMemo(() => {
    const x = Array.from({ length: epochs }, (_, i) => i + 1);
    const train = x.map((e) => {
      const base = Math.exp(-e / 11);
      const noise = 0.03 * noiseFactor * Math.sin(e / 2 + trainPct * 0.1);
      return base + noise;
    });
    const val = x.map((e) => {
      const base = Math.exp(-e / 10);
      const penalty = e > overfitStart ? 0.003 * noiseFactor * (e - overfitStart) ** 2 : 0;
      const noise = 0.04 * noiseFactor * Math.cos(e / 3 + valPct * 0.1);
      return base + 0.07 * noiseFactor + penalty + noise;
    });
    const test = x.map((e) => {
      const base = Math.exp(-e / 10);
      const penalty = e > overfitStart ? 0.004 * noiseFactor * (e - overfitStart) ** 2 : 0;
      return base + 0.12 * noiseFactor + penalty;
    });
    return { x, train, val, test };
  }, [epochs, trainPct, valPct, overfitStart, noiseFactor]);

  return (
    <SimulationPanel title="Train / Validation / Test Split">
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
          <div>
            <SimulationLabel>Train: {trainPct}%</SimulationLabel>
            <Slider
              min={30}
              max={85}
              step={5}
              value={[trainPct]}
              onValueChange={([v]) => {
                const remaining = 100 - v;
                setTrainPct(v);
                if (valPct > remaining - 5) setValPct(Math.max(5, remaining - 5));
              }}
            />
          </div>
          <div>
            <SimulationLabel>Val: {valPct}%</SimulationLabel>
            <Slider
              min={5}
              max={Math.max(5, 100 - trainPct - 5)}
              step={5}
              value={[valPct]}
              onValueChange={([v]) => setValPct(v)}
            />
          </div>
          <div className="flex items-end">
            <span className="rounded bg-[var(--surface-2,#27272a)] px-3 py-1.5 text-sm text-[var(--text-strong)]">
              Test: {testPct}%
            </span>
          </div>
          <div>
            <SimulationLabel>Epochs: {epochs}</SimulationLabel>
            <Slider min={15} max={80} step={1} value={[epochs]} onValueChange={([v]) => setEpochs(v)} />
          </div>
        </div>
        {/* Visual split bar */}
        <div className="flex h-6 w-full overflow-hidden rounded-full text-xs font-medium">
          <div className="flex items-center justify-center bg-blue-600 text-white" style={{ width: `${trainPct}%` }}>
            Train
          </div>
          <div className="flex items-center justify-center bg-amber-500 text-white" style={{ width: `${valPct}%` }}>
            Val
          </div>
          <div className="flex items-center justify-center bg-emerald-600 text-white" style={{ width: `${testPct}%` }}>
            Test
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
        data={[
          { x: curves.x, y: curves.train, type: 'scatter', mode: 'lines', name: 'Training loss', line: { color: '#3b82f6', width: 2 } },
          { x: curves.x, y: curves.val, type: 'scatter', mode: 'lines', name: 'Validation loss', line: { color: '#f59e0b', width: 2 } },
          { x: curves.x, y: curves.test, type: 'scatter', mode: 'lines', name: 'Test loss', line: { color: '#10b981', width: 2 } },
          // Early stopping line
          { x: [overfitStart, overfitStart], y: [0, 1.2], type: 'scatter', mode: 'lines', name: 'Early stop', line: { color: '#ef4444', width: 1 } },
        ]}
        layout={{
          xaxis: { title: { text: 'Epoch' } },
          yaxis: { title: { text: 'Loss' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 360 }}
      />
        {trainPct < 45 && (
          <div className="mt-2 text-sm text-amber-400">
            With only {trainPct}% training data, curves are noisy and overfitting starts earlier.
          </div>
        )}
      </SimulationMain>
    </SimulationPanel>
  );
}
