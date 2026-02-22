"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationCheckbox } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateCurves(dropout: number, weightDecay: number, seed: number) {
  const rng = mulberry32(seed);
  const epochs = 100;
  const trainLoss: number[] = [];
  const valLoss: number[] = [];

  // Regularization strength affects the gap and minimum
  const regStrength = dropout * 0.5 + weightDecay * 10;
  const baseDecay = 0.04 + regStrength * 0.01;
  const overfitRate = Math.max(0, 0.015 - regStrength * 0.008);
  const noise = 0.02;

  for (let e = 0; e < epochs; e++) {
    const t = e / epochs;
    // Training loss always decreases
    const tl = 2.0 * Math.exp(-baseDecay * e) + 0.05 + (rng() - 0.5) * noise * 0.3;
    // Validation loss: decreases then may increase (overfit)
    const vl = 2.0 * Math.exp(-baseDecay * 0.8 * e) + 0.15 + overfitRate * e * t + (rng() - 0.5) * noise;
    trainLoss.push(Math.max(0.02, tl));
    valLoss.push(Math.max(0.05, vl));
  }

  return { trainLoss, valLoss };
}

export default function RegularizationEffects({}: SimulationComponentProps) {
  const [dropout, setDropout] = useState(0);
  const [weightDecay, setWeightDecay] = useState(0);
  const [showNoReg, setShowNoReg] = useState(true);

  const noRegCurves = useMemo(() => generateCurves(0, 0, 42), []);
  const regCurves = useMemo(() => generateCurves(dropout, weightDecay, 43), [dropout, weightDecay]);

  const epochs = Array.from({ length: 100 }, (_, i) => i);

  const traces: any[] = [];

  if (showNoReg) {
    traces.push({
      type: 'scatter',
      x: epochs,
      y: noRegCurves.trainLoss,
      mode: 'lines',
      line: { color: '#ef4444', width: 2, dash: 'dash' },
      name: 'Train (no reg)',
    });
    traces.push({
      type: 'scatter',
      x: epochs,
      y: noRegCurves.valLoss,
      mode: 'lines',
      line: { color: '#f97316', width: 2, dash: 'dash' },
      name: 'Val (no reg)',
    });
  }

  traces.push({
    type: 'scatter',
    x: epochs,
    y: regCurves.trainLoss,
    mode: 'lines',
    line: { color: '#3b82f6', width: 2.5 },
    name: `Train (dropout=${dropout}, wd=${weightDecay})`,
  });
  traces.push({
    type: 'scatter',
    x: epochs,
    y: regCurves.valLoss,
    mode: 'lines',
    line: { color: '#10b981', width: 2.5 },
    name: `Val (dropout=${dropout}, wd=${weightDecay})`,
  });

  const gap = regCurves.valLoss[99] - regCurves.trainLoss[99];
  const noRegGap = noRegCurves.valLoss[99] - noRegCurves.trainLoss[99];

  return (
    <SimulationPanel title="Regularization Effects on Training">
      <SimulationSettings>
        <SimulationCheckbox checked={showNoReg} onChange={setShowNoReg} label="Show no-regularization baseline" />
      </SimulationSettings>
      <SimulationConfig>
        <div className="space-y-4">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Dropout rate: {dropout.toFixed(2)}
            </SimulationLabel>
            <Slider min={0} max={0.5} step={0.05} value={[dropout]} onValueChange={([v]) => setDropout(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Weight decay: {weightDecay.toFixed(3)}
            </SimulationLabel>
            <Slider min={0} max={0.05} step={0.005} value={[weightDecay]} onValueChange={([v]) => setWeightDecay(v)} className="w-full" />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
          data={traces}
          layout={{
            xaxis: { title: { text: 'Epoch' } },
            yaxis: { title: { text: 'Loss' } },
            margin: { t: 30, b: 50, l: 60, r: 30 },
            autosize: true,
          }}
          style={{ width: '100%', height: '400px' }}
        />
      </SimulationMain>
      <SimulationResults>
        <div className="p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)] space-y-2">
          <p><strong className="text-[var(--text-strong)]">Generalization gap</strong> (train-val difference at epoch 100):</p>
          {showNoReg && (
            <p>No regularization: <span className="text-orange-400">{noRegGap.toFixed(3)}</span></p>
          )}
          <p>With regularization: <span className="text-green-400">{gap.toFixed(3)}</span></p>
          {gap < noRegGap * 0.7 && (
            <p className="text-blue-400">Regularization is significantly reducing overfitting.</p>
          )}
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
