"use client";

import { useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function ActivationFunctions({}: SimulationComponentProps) {
  const xs = useMemo(() => Array.from({ length: 241 }, (_, i) => -6 + i * 0.05), []);

  const relu = useMemo(() => xs.map((x) => Math.max(0, x)), [xs]);
  const sigmoid = useMemo(() => xs.map((x) => 1 / (1 + Math.exp(-x))), [xs]);
  const tanh = useMemo(() => xs.map((x) => Math.tanh(x)), [xs]);
  const softmax = useMemo(() => {
    const exps = xs.map((x) => Math.exp(x));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }, [xs]);

  return (
    <SimulationPanel title="Activation Functions" caption="Comparison of ReLU, Sigmoid, Tanh, and Softmax over a one-dimensional input grid.">
      <SimulationMain>
      <CanvasChart
        data={[
          { x: xs, y: relu, type: 'scatter', mode: 'lines', name: 'ReLU', line: { color: '#60a5fa', width: 2 } },
          { x: xs, y: sigmoid, type: 'scatter', mode: 'lines', name: 'Sigmoid', line: { color: '#4ade80', width: 2 } },
          { x: xs, y: tanh, type: 'scatter', mode: 'lines', name: 'Tanh', line: { color: '#facc15', width: 2 } },
          { x: xs, y: softmax, type: 'scatter', mode: 'lines', name: 'Softmax (vector-normalized)', line: { color: '#f87171', width: 2 } },
        ]}
        layout={{
          title: { text: 'Common nonlinearities' },
          xaxis: { title: { text: 'x' } },
          yaxis: { title: { text: 'f(x)' } },
          height: 420,
          margin: { t: 40, b: 60, l: 60, r: 20 },
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
