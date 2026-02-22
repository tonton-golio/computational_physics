"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationCheckbox } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function computeDoubleDescent(numSamples: number, peakSharpness: number) {
  const modelSizes: number[] = [];
  const testError: number[] = [];
  const trainError: number[] = [];
  const classicalBV: number[] = [];

  // Interpolation threshold: roughly when params = samples
  const threshold = numSamples;

  for (let p = 5; p <= 500; p += 2) {
    modelSizes.push(p);

    // Classical bias-variance
    const bias = 2.0 / (1 + p * 0.02);
    const variance = 0.01 * p / numSamples;
    classicalBV.push(bias + variance);

    // Training error
    if (p < threshold) {
      trainError.push(Math.max(0, 0.5 * (1 - p / threshold)));
    } else {
      trainError.push(0);
    }

    // Double descent test error
    const ratio = p / threshold;
    if (ratio < 0.95) {
      // Underparameterized: classical U-shape
      const b = 1.5 / (1 + p * 0.015);
      const v = 0.008 * p / numSamples;
      testError.push(b + v);
    } else if (ratio < 1.1) {
      // Near interpolation threshold: peak
      const peakHeight = peakSharpness * 2.0;
      const dist = Math.abs(ratio - 1.0);
      testError.push(0.4 + peakHeight * Math.exp(-dist * dist / 0.003));
    } else {
      // Overparameterized: second descent
      const overParam = ratio - 1;
      testError.push(0.3 * Math.exp(-overParam * 0.8) + 0.12);
    }
  }

  return { modelSizes, testError, trainError, classicalBV };
}

export default function DoubleDescent({}: SimulationComponentProps) {
  const [numSamples, setNumSamples] = useState(100);
  const [peakSharpness, setPeakSharpness] = useState(1.0);
  const [showClassical, setShowClassical] = useState(true);

  const data = useMemo(() => computeDoubleDescent(numSamples, peakSharpness), [numSamples, peakSharpness]);

  const traces: any[] = [
    {
      type: 'scatter',
      x: data.modelSizes,
      y: data.testError,
      mode: 'lines',
      line: { color: '#3b82f6', width: 3 },
      name: 'Test error (double descent)',
    },
    {
      type: 'scatter',
      x: data.modelSizes,
      y: data.trainError,
      mode: 'lines',
      line: { color: '#10b981', width: 2, dash: 'dot' },
      name: 'Train error',
    },
  ];

  if (showClassical) {
    traces.push({
      type: 'scatter',
      x: data.modelSizes,
      y: data.classicalBV,
      mode: 'lines',
      line: { color: '#f97316', width: 2, dash: 'dash' },
      name: 'Classical bias-variance',
    });
  }

  // Interpolation threshold line
  const shapes: any[] = [
    {
      type: 'line',
      x0: numSamples,
      x1: numSamples,
      y0: 0,
      y1: 3,
      line: { color: '#ef4444', width: 1.5, dash: 'dashdot' },
    },
  ];

  return (
    <SimulationPanel title="Double Descent Phenomenon">
      <SimulationSettings>
        <SimulationCheckbox checked={showClassical} onChange={setShowClassical} label="Show classical bias-variance" />

        <div className="p-3 bg-[var(--surface-2)] rounded text-xs text-[var(--text-muted)] space-y-2">
          <p>The <span className="text-red-400">red dashed line</span> marks the interpolation threshold (params = samples).</p>
          <p><strong>Left of threshold:</strong> Classical regime where adding parameters first reduces bias, then increases variance.</p>
          <p><strong>At threshold:</strong> The model barely fits the data exactly, amplifying noise and creating the error peak.</p>
          <p><strong>Right of threshold:</strong> Overparameterized models find smoother interpolations, decreasing test error again.</p>
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="space-y-4">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Number of samples: {numSamples}
            </SimulationLabel>
            <Slider min={30} max={250} step={10} value={[numSamples]} onValueChange={([v]) => setNumSamples(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Peak sharpness: {peakSharpness.toFixed(1)}
            </SimulationLabel>
            <Slider min={0.1} max={3} step={0.1} value={[peakSharpness]} onValueChange={([v]) => setPeakSharpness(v)} className="w-full" />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
          data={traces}
          layout={{
            xaxis: { title: { text: 'Number of parameters' } },
            yaxis: { title: { text: 'Error' }, range: [0, 2.5] },
            shapes,
            margin: { t: 30, b: 50, l: 60, r: 30 },
            autosize: true,
            annotations: [{
              x: numSamples,
              y: 2.3,
              text: 'Interpolation<br>threshold',
              showarrow: false,
              font: { color: '#ef4444', size: 10 },
            }],
          }}
          style={{ width: '100%', height: '450px' }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
