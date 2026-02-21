'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

/**
 * Gillespie (Stochastic Simulation Algorithm) for a birth-death process.
 *
 * Birth:  0 -> X  at rate k
 * Death:  X -> 0  at rate gamma * N
 *
 * The ODE mean is: dN/dt = k - gamma * N, steady state N_ss = k / gamma.
 *
 * Multiple stochastic trajectories are plotted alongside the deterministic ODE solution.
 */

function gillespieTrajectory(
  k: number, gamma: number, tMax: number, seed: number,
): { times: number[]; counts: number[] } {
  let state = seed;
  const rand = () => {
    state = (state * 1664525 + 1013904223) & 0x7fffffff;
    return Math.max(1e-15, state / 0x7fffffff);
  };

  let N = 0;
  let t = 0;
  const times: number[] = [0];
  const counts: number[] = [0];

  while (t < tMax) {
    const birthRate = k;
    const deathRate = gamma * N;
    const totalRate = birthRate + deathRate;

    if (totalRate <= 0) {
      // Only births possible
      const dt = -Math.log(rand()) / k;
      t += dt;
      if (t >= tMax) break;
      N += 1;
    } else {
      const dt = -Math.log(rand()) / totalRate;
      t += dt;
      if (t >= tMax) break;

      // Decide which reaction fires
      if (rand() < birthRate / totalRate) {
        N += 1;
      } else {
        N = Math.max(0, N - 1);
      }
    }

    times.push(t);
    counts.push(N);
  }

  // Add final point at tMax
  times.push(tMax);
  counts.push(N);

  return { times, counts };
}

function subsampleTrajectory(
  times: number[], counts: number[], nPoints: number,
): { t: number[]; n: number[] } {
  const tMax = times[times.length - 1];
  const tSampled: number[] = [];
  const nSampled: number[] = [];
  let idx = 0;

  for (let i = 0; i <= nPoints; i++) {
    const target = (i / nPoints) * tMax;
    while (idx < times.length - 1 && times[idx + 1] <= target) {
      idx++;
    }
    tSampled.push(target);
    nSampled.push(counts[idx]);
  }

  return { t: tSampled, n: nSampled };
}

export default function GillespieTrajectory() {
  const [k, setK] = useState(20);
  const [gamma, setGamma] = useState(0.5);
  const [numTrajectories, setNumTrajectories] = useState(5);
  const [seed, setSeed] = useState(42);

  const nSS = k / gamma;
  const tMax = Math.max(5 / gamma, 10);

  const { stochasticTraces, odeTrace } = useMemo(() => {
    const nDisplay = 400;

    // ODE solution
    const odeT: number[] = [];
    const odeN: number[] = [];
    for (let i = 0; i <= nDisplay; i++) {
      const t = (i / nDisplay) * tMax;
      odeT.push(t);
      odeN.push(nSS * (1 - Math.exp(-gamma * t)));
    }

    const colors = ['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#ec4899', '#06b6d4', '#84cc16', '#f43f5e'];
    const stochasticTraces: any[] = [];

    for (let j = 0; j < numTrajectories; j++) {
      const traj = gillespieTrajectory(k, gamma, tMax, seed + j * 137);
      const sub = subsampleTrajectory(traj.times, traj.counts, nDisplay);
      stochasticTraces.push({
        x: sub.t,
        y: sub.n,
        type: 'scatter',
        mode: 'lines',
        line: { color: colors[j % colors.length], width: 1.5 },
        name: `Trajectory ${j + 1}`,
        opacity: 0.7,
      });
    }

    const odeTrace = {
      x: odeT,
      y: odeN,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#ffffff', width: 3 },
      name: 'ODE mean',
    };

    return { stochasticTraces, odeTrace };
  }, [k, gamma, numTrajectories, seed, nSS, tMax]);

  // Steady-state dashed line
  const ssLine = {
    x: [0, tMax],
    y: [nSS, nSS],
    type: 'scatter',
    mode: 'lines',
    line: { color: '#9ca3af', width: 1.5, dash: 'dash' },
    name: `n_ss = ${nSS.toFixed(0)}`,
  };

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Gillespie Algorithm: Stochastic Birth-Death Process
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Production rate k: {k}
          </label>
          <Slider value={[k]} onValueChange={([v]) => setK(v)} min={1} max={100} step={1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Degradation rate &Gamma;: {gamma.toFixed(2)}
          </label>
          <Slider value={[gamma]} onValueChange={([v]) => setGamma(v)} min={0.05} max={2.0} step={0.05} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Trajectories: {numTrajectories}
          </label>
          <Slider value={[numTrajectories]} onValueChange={([v]) => setNumTrajectories(v)} min={1} max={8} step={1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Seed: {seed}
          </label>
          <Slider value={[seed]} onValueChange={([v]) => setSeed(v)} min={1} max={100} step={1} />
        </div>
      </div>

      <CanvasChart
        data={[ssLine, ...stochasticTraces, odeTrace] as any}
        layout={{
          height: 420,
          margin: { t: 20, b: 55, l: 55, r: 20 },
          xaxis: {
            title: { text: 'Time' },
            range: [0, tMax],
          },
          yaxis: {
            title: { text: 'Molecule count N' },
            range: [0, nSS * 2.5],
          },
          showlegend: true,
        }}
        style={{ width: '100%' }}
      />

      <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Steady state</div>
          <div className="text-lg font-mono">{nSS.toFixed(1)}</div>
          <div className="text-xs">k / &Gamma;</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Expected CV</div>
          <div className="text-lg font-mono">{(1 / Math.sqrt(nSS)).toFixed(3)}</div>
          <div className="text-xs">1/&radic;n_ss (Poisson)</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Fano factor</div>
          <div className="text-lg font-mono">1.0</div>
          <div className="text-xs">Var/Mean (Poisson)</div>
        </div>
      </div>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          The white curve is the deterministic ODE solution. The colored traces are
          individual stochastic realizations &mdash; each one is a single cell&apos;s molecule
          count over time. With large N (high k, low &Gamma;), the traces cluster tightly
          around the ODE mean. With small N (low k), the noise is dramatic &mdash; individual
          trajectories wander far from the mean. Try setting k = 5 vs k = 100 to see the
          small-number effect in action.
        </p>
      </div>
    </div>
  );
}
