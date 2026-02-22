"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function HillFunction({}: SimulationComponentProps) {
  const [K, setK] = useState(1.0);
  const [H, setH] = useState(1);

  const { xVals, yRepression, yActivation } = useMemo(() => {
    const n = 500;
    const xMax = 3.0;
    const xVals: number[] = [];
    const yRepression: number[] = [];
    const yActivation: number[] = [];

    for (let i = 0; i <= n; i++) {
      const x = (i / n) * xMax;
      xVals.push(x);
      const xOverK_H = Math.pow(x / K, H);
      yRepression.push(1.0 / (1.0 + xOverK_H));
      yActivation.push(xOverK_H / (1.0 + xOverK_H));
    }

    return { xVals, yRepression, yActivation };
  }, [K, H]);

  const commonLayout = {
    height: 380,
    margin: { t: 40, b: 50, l: 50, r: 20 },
    xaxis: {
      title: { text: 'Concentration of TF' },
      range: [0, 3],
    },
    yaxis: {
      title: { text: 'Hill function value' },
      range: [0, 1.05],
    },
  };

  return (
    <SimulationPanel title="Hill Function: Repression and Activation">
      <SimulationConfig>
        <div>
          <SimulationLabel>Dissociation constant K: {K.toFixed(2)}</SimulationLabel>
          <Slider
            min={0.1}
            max={2.5}
            step={0.05}
            value={[K]}
            onValueChange={([v]) => setK(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>Hill coefficient H: {H}</SimulationLabel>
          <Slider
            min={1}
            max={10}
            step={1}
            value={[H]}
            onValueChange={([v]) => setH(v)}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <CanvasChart
            data={[
              {
                x: xVals,
                y: yRepression,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#f97316', width: 2.5 },
                name: 'Repression',
              },
              {
                x: [0, 3],
                y: [0.5, 0.5],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#6b7280', dash: 'dash', width: 1 },
                showlegend: false,
              },
              {
                x: [K, K],
                y: [0, 1],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#6b7280', dash: 'dash', width: 1 },
                showlegend: false,
              },
            ] as any}
            layout={{
              ...commonLayout,
              title: { text: 'Repression: K^H / (K^H + x^H)', font: { size: 14 } },
            }}
            style={{ width: '100%' }}
          />
        </div>
        <div>
          <CanvasChart
            data={[
              {
                x: xVals,
                y: yActivation,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#3b82f6', width: 2.5 },
                name: 'Activation',
              },
              {
                x: [0, 3],
                y: [0.5, 0.5],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#6b7280', dash: 'dash', width: 1 },
                showlegend: false,
              },
              {
                x: [K, K],
                y: [0, 1],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#6b7280', dash: 'dash', width: 1 },
                showlegend: false,
              },
            ] as any}
            layout={{
              ...commonLayout,
              title: { text: 'Activation: x^H / (K^H + x^H)', font: { size: 14 } },
            }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
      </SimulationMain>

      <div className="mt-4 text-sm text-[var(--text-muted)]">
        <p>
          The <strong className="text-[var(--text-muted)]">Hill function</strong> models cooperative binding of transcription factors to promoter regions.
          The dissociation constant <em>K</em> sets the threshold concentration at which the function reaches half its maximum value.
          The Hill coefficient <em>H</em> controls the steepness of the response: higher values create a more switch-like behavior.
        </p>
      </div>
    </SimulationPanel>
  );
}
