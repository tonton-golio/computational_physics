'use client';

import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function FTLInstability({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const rounds = 200;
  const t = useMemo(() => Array.from({ length: rounds }, (_, i) => i + 1), []);
  const actions = useMemo(() => t.map((x) => (x % 2 === 0 ? 1 : 0)), [t]);
  const ftlLoss = useMemo(() => t.map((x) => x * 0.5 + (x % 2 === 0 ? 6 : 1)), [t]);
  const regularizedLoss = useMemo(() => t.map((x) => 0.26 * x + 4 * Math.sqrt(x)), [t]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">FTL Instability Example</h3>
      <p className="text-sm text-gray-400 mb-4">
        In alternating adversarial losses, FTL can chase the last winner and oscillate.
      </p>
      <Plot
        data={[
          { x: t, y: actions.map((a) => a * 4), type: 'scatter', mode: 'lines', name: 'FTL action (scaled)', line: { color: '#c084fc', width: 1 } },
          { x: t, y: ftlLoss, type: 'scatter', mode: 'lines', name: 'FTL cumulative loss', line: { color: '#f87171', width: 2 } },
          { x: t, y: regularizedLoss, type: 'scatter', mode: 'lines', name: 'Regularized baseline', line: { color: '#4ade80', width: 2 } },
        ]}
        layout={{
          title: { text: 'Oscillation under adversarial alternation' },
          xaxis: { title: { text: 'round t' }, color: '#9ca3af' },
          yaxis: { title: { text: 'cumulative loss' }, color: '#9ca3af' },
          height: 420,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
