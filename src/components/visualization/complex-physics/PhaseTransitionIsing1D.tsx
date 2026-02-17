'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function transferMatrixMagnetization(beta: number, h: number, J = 1): number {
  const a = Math.exp(beta * J + beta * h);
  const b = Math.exp(-beta * J);
  const c = Math.exp(beta * J - beta * h);
  const trace = a + c;
  const det = a * c - b * b;
  const disc = Math.sqrt(Math.max(trace * trace - 4 * det, 0));
  const lambda = 0.5 * (trace + disc);
  const dLambdaDh =
    0.5 *
    (beta * (a - c) +
      (trace * beta * (a - c) - 2 * beta * (a * c + b * b)) / (disc || 1e-10));
  return dLambdaDh / (beta * lambda);
}

function runMonteCarlo1D(size: number, beta: number, nsteps: number, runSeed: number): number[] {
  const rand = mulberry32((size * 1009 + Math.floor(beta * 1000) * 9176 + runSeed * 131) >>> 0);
  const spins = Array.from({ length: size }, () => (rand() > 0.5 ? 1 : -1));
  const magnetization: number[] = [];
  let m = spins.reduce((s, v) => s + v, 0);
  for (let step = 0; step < nsteps; step++) {
    const i = Math.floor(rand() * size);
    const left = spins[(i - 1 + size) % size];
    const right = spins[(i + 1) % size];
    const dE = 2 * spins[i] * (left + right);
    if (dE <= 0 || rand() < Math.exp(-beta * dE)) {
      spins[i] *= -1;
      m += 2 * spins[i];
    }
    magnetization.push(Math.abs(m / size));
  }
  return magnetization;
}

export function PhaseTransitionIsing1D() {
  const [size, setSize] = useState(60);
  const [beta, setBeta] = useState(1.2);
  const [nsteps, setNsteps] = useState(3000);
  const [rerun, setRerun] = useState(0);

  const mcSeries = useMemo(() => runMonteCarlo1D(size, beta, nsteps, rerun), [size, beta, nsteps, rerun]);

  const betaSweep = useMemo(() => {
    const betas: number[] = [];
    const mExact: number[] = [];
    for (let b = 0.1; b <= 3.0; b += 0.05) {
      betas.push(Number(b.toFixed(2)));
      mExact.push(Math.abs(transferMatrixMagnetization(b, 0.1)));
    }
    return { betas, mExact };
  }, []);

  const darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#9ca3af' },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    xaxis: { gridcolor: '#1e1e2e' },
    yaxis: { gridcolor: '#1e1e2e' },
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Size: {size}</label>
          <input type="range" min={20} max={200} step={10} value={size} onChange={e => setSize(Number(e.target.value))} className="w-48" />
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Beta (1/T): {beta.toFixed(2)}</label>
          <input type="range" min={0.1} max={3} step={0.05} value={beta} onChange={e => setBeta(Number(e.target.value))} className="w-48" />
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Steps: {nsteps}</label>
          <input type="range" min={500} max={12000} step={500} value={nsteps} onChange={e => setNsteps(Number(e.target.value))} className="w-48" />
        </div>
        <button onClick={() => setRerun(v => v + 1)} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm mt-4">
          Re-run
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[
            { y: mcSeries, type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 1.5 }, name: '|m| Monte Carlo' },
          ]}
          layout={{
            ...darkLayout,
            title: { text: '1D Ising Monte Carlo: |m| vs Step', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'Step' } },
            yaxis: { ...darkLayout.yaxis, title: { text: '|m|' } },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
        <Plot
          data={[
            { x: betaSweep.betas, y: betaSweep.mExact, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2 }, name: 'Transfer matrix (h=0.1)' },
          ]}
          layout={{
            ...darkLayout,
            title: { text: '1D Ising (Transfer Matrix) Magnetization', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'Beta (1/T)' } },
            yaxis: { ...darkLayout.yaxis, title: { text: '|m|' } },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
      </div>
    </div>
  );
}
