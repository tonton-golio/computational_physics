'use client';

import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

export default function ActivationFunctions({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
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
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">Activation Functions</h3>
      <p className="text-sm text-gray-400 mb-4">
        Comparison of ReLU, Sigmoid, Tanh, and Softmax over a one-dimensional input grid.
      </p>
      <Plot
        data={[
          { x: xs, y: relu, type: 'scatter', mode: 'lines', name: 'ReLU', line: { color: '#60a5fa', width: 2 } },
          { x: xs, y: sigmoid, type: 'scatter', mode: 'lines', name: 'Sigmoid', line: { color: '#4ade80', width: 2 } },
          { x: xs, y: tanh, type: 'scatter', mode: 'lines', name: 'Tanh', line: { color: '#facc15', width: 2 } },
          { x: xs, y: softmax, type: 'scatter', mode: 'lines', name: 'Softmax (vector-normalized)', line: { color: '#f87171', width: 2 } },
        ]}
        layout={{
          title: { text: 'Common nonlinearities' },
          xaxis: { title: { text: 'x' }, color: '#9ca3af' },
          yaxis: { title: { text: 'f(x)' }, color: '#9ca3af' },
          height: 420,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          margin: { t: 40, b: 60, l: 60, r: 20 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
