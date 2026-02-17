'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function StochasticApproximation({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [steps, setSteps] = useState(300);
  const [alpha, setAlpha] = useState(0.03);
  const [noiseScale, setNoiseScale] = useState(0.1);
  const [seed, setSeed] = useState(7);

  const { xs, ys } = useMemo(() => {
    const rng = mulberry32(seed);
    const xVals: number[] = [0.6];
    const yVals: number[] = [Math.abs(xVals[0] * xVals[0] - 1)];
    for (let k = 1; k <= steps; k++) {
      const prev = xVals[k - 1];
      const noise = (rng() - 0.5) * 2 * noiseScale;
      const gradientProxy = (prev * prev - 1) + noise;
      const next = prev - alpha * gradientProxy;
      xVals.push(next);
      yVals.push(Math.abs(next * next - 1));
    }
    return { xs: xVals, ys: yVals };
  }, [steps, alpha, noiseScale, seed]);

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">Robbins-Monro Stochastic Approximation</h3>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
        <label className="text-xs text-gray-300">Steps: {steps}<input className="w-full" type="range" min={50} max={1000} step={10} value={steps} onChange={(e) => setSteps(parseInt(e.target.value))} /></label>
        <label className="text-xs text-gray-300">Step size: {alpha.toFixed(3)}<input className="w-full" type="range" min={0.005} max={0.1} step={0.005} value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))} /></label>
        <label className="text-xs text-gray-300">Noise: {noiseScale.toFixed(2)}<input className="w-full" type="range" min={0} max={0.5} step={0.01} value={noiseScale} onChange={(e) => setNoiseScale(parseFloat(e.target.value))} /></label>
        <button onClick={() => setSeed((s) => s + 1)} className="rounded bg-blue-600 text-white text-sm px-3 py-2">Re-sample</button>
      </div>
      <Plot
        data={[
          { x: Array.from({ length: xs.length }, (_, i) => i), y: xs, type: 'scatter', mode: 'lines', name: 'x_k', line: { color: '#60a5fa', width: 2 } },
          { x: Array.from({ length: ys.length }, (_, i) => i), y: ys.map((v) => Math.log10(v + 1e-6)), type: 'scatter', mode: 'lines', name: 'log10 residual', line: { color: '#facc15', width: 2 } },
        ]}
        layout={{
          title: { text: 'Convergence behavior under noisy updates' },
          xaxis: { title: { text: 'iteration k' }, color: '#9ca3af' },
          yaxis: { title: { text: 'state / log residual' }, color: '#9ca3af' },
          height: 420,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          margin: { t: 40, b: 60, l: 60, r: 60 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
