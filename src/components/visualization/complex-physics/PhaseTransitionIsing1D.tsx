'use client';

import React, { useMemo, useState } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';

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

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Size: {size}</label>
          <Slider value={[size]} onValueChange={([v]) => setSize(v)} min={20} max={200} step={10} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Beta (1/T): {beta.toFixed(2)}</label>
          <Slider value={[beta]} onValueChange={([v]) => setBeta(v)} min={0.1} max={3} step={0.05} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Steps: {nsteps}</label>
          <Slider value={[nsteps]} onValueChange={([v]) => setNsteps(v)} min={500} max={12000} step={500} />
        </div>
        <button onClick={() => setRerun(v => v + 1)} className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4">
          Re-run
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CanvasChart
          data={[
            { y: mcSeries, type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 1.5 }, name: '|m| Monte Carlo' },
          ]}
          layout={{
            title: { text: '1D Ising Monte Carlo: |m| vs Step', font: { size: 13 } },
            xaxis: { title: { text: 'Step' } },
            yaxis: { title: { text: '|m|' } },
            showlegend: false,
            margin: { t: 40, r: 20, b: 50, l: 60 },
          }}
          style={{ width: '100%', height: 320 }}
        />
        <CanvasChart
          data={[
            { x: betaSweep.betas, y: betaSweep.mExact, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2 }, name: 'Transfer matrix (h=0.1)' },
          ]}
          layout={{
            title: { text: '1D Ising (Transfer Matrix) Magnetization', font: { size: 13 } },
            xaxis: { title: { text: 'Beta (1/T)' } },
            yaxis: { title: { text: '|m|' } },
            showlegend: false,
            margin: { t: 40, r: 20, b: 50, l: 60 },
          }}
          style={{ width: '100%', height: 320 }}
        />
      </div>
    </div>
  );
}
