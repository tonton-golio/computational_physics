'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const ROWS = 4, COLS = 12;
const START = [3, 0], GOAL = [3, 11];
const ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right

function isCliff(r: number, c: number) { return r === 3 && c > 0 && c < 11; }

function step(r: number, c: number, a: number): [number, number, number] {
  const nr = Math.max(0, Math.min(ROWS - 1, r + ACTIONS[a][0]));
  const nc = Math.max(0, Math.min(COLS - 1, c + ACTIONS[a][1]));
  if (isCliff(nr, nc)) return [START[0], START[1], -100];
  return [nr, nc, -1];
}

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function epsGreedy(Q: Float64Array, s: number, eps: number, rng: () => number): number {
  if (rng() < eps) return Math.floor(rng() * 4);
  let best = 0;
  for (let a = 1; a < 4; a++) if (Q[s * 4 + a] > Q[s * 4 + best]) best = a;
  return best;
}

function train(algo: 'sarsa' | 'qlearning', eps: number, numEp: number, seed: number) {
  const rng = mulberry32(seed);
  const Q = new Float64Array(ROWS * COLS * 4);
  const alpha = 0.5, gamma = 1.0;
  const rewards: number[] = [];

  for (let ep = 0; ep < numEp; ep++) {
    let r = START[0], c = START[1], totalR = 0;
    const si = r * COLS + c;
    let a = epsGreedy(Q, si, eps, rng);
    for (let t = 0; t < 500; t++) {
      const [nr, nc, reward] = step(r, c, a);
      totalR += reward;
      const nsi = nr * COLS + nc;
      if (algo === 'sarsa') {
        const na = (nr === GOAL[0] && nc === GOAL[1]) ? 0 : epsGreedy(Q, nsi, eps, rng);
        Q[si * 4 + a] += alpha * (reward + gamma * Q[nsi * 4 + na] - Q[si * 4 + a]);
        r = nr; c = nc; a = na;
      } else {
        let maxQ = Q[nsi * 4];
        for (let b = 1; b < 4; b++) if (Q[nsi * 4 + b] > maxQ) maxQ = Q[nsi * 4 + b];
        Q[si * 4 + a] += alpha * (reward + gamma * maxQ - Q[si * 4 + a]);
        r = nr; c = nc;
        a = epsGreedy(Q, r * COLS + c, eps, rng);
      }
      if (r === GOAL[0] && c === GOAL[1]) break;
    }
    rewards.push(totalR);
  }

  // Extract greedy path
  const path: [number, number][] = [[START[0], START[1]]];
  let pr = START[0], pc = START[1];
  for (let t = 0; t < 50; t++) {
    if (pr === GOAL[0] && pc === GOAL[1]) break;
    const psi = pr * COLS + pc;
    let best = 0;
    for (let a = 1; a < 4; a++) if (Q[psi * 4 + a] > Q[psi * 4 + best]) best = a;
    const [nr, nc] = step(pr, pc, best);
    if (nr === START[0] && nc === START[1] && isCliff(pr + ACTIONS[best][0], pc + ACTIONS[best][1])) break;
    pr = nr; pc = nc;
    path.push([pr, pc]);
  }
  return { rewards, path };
}

function smooth(arr: number[], w: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    const lo = Math.max(0, i - w + 1);
    let sum = 0;
    for (let j = lo; j <= i; j++) sum += arr[j];
    out.push(sum / (i - lo + 1));
  }
  return out;
}

export default function CliffWalking({ id }: SimulationComponentProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [epsilon, setEpsilon] = useState(0.1);
  const [numEpisodes, setNumEpisodes] = useState(300);
  const [seed, setSeed] = useState(7);

  const { rewardTraces, pathTraces } = useMemo(() => {
    const sarsa = train('sarsa', epsilon, numEpisodes, seed);
    const ql = train('qlearning', epsilon, numEpisodes, seed + 1000);
    const x = Array.from({ length: numEpisodes }, (_, i) => i + 1);
    const sSmooth = smooth(sarsa.rewards, 20);
    const qSmooth = smooth(ql.rewards, 20);

    const rewardTraces: ChartTrace[] = [
      { x, y: sSmooth, type: 'scatter', mode: 'lines', name: 'SARSA (safe path)', line: { color: '#facc15', width: 2 } },
      { x, y: qSmooth, type: 'scatter', mode: 'lines', name: 'Q-learning (optimal path)', line: { color: '#4ade80', width: 2 } },
    ];

    const pathTraces: ChartTrace[] = [
      // cliff markers
      { x: Array.from({ length: 10 }, (_, i) => i + 1), y: Array(10).fill(3),
        type: 'scatter', mode: 'markers', name: 'Cliff',
        marker: { color: '#ef4444', size: 10, symbol: 'square' }, showlegend: true },
      // SARSA path
      { x: sarsa.path.map(p => p[1]), y: sarsa.path.map(p => p[0]),
        type: 'scatter', mode: 'lines+markers', name: 'SARSA path',
        line: { color: '#facc15', width: 2.5 }, marker: { size: 5 } },
      // Q-learning path
      { x: ql.path.map(p => p[1]), y: ql.path.map(p => p[0]),
        type: 'scatter', mode: 'lines+markers', name: 'Q-learning path',
        line: { color: '#4ade80', width: 2.5 }, marker: { size: 5 } },
    ];
    return { rewardTraces, pathTraces };
  }, [epsilon, numEpisodes, seed]);

  const handleResample = useCallback(() => setSeed((s) => s + 1), []);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">
        Cliff Walking: SARSA vs Q-Learning
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        SARSA learns a safe path away from the cliff (accounting for its own exploration),
        while Q-learning finds the optimal path along the cliff edge.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Epsilon: {epsilon.toFixed(2)}</label>
          <Slider value={[epsilon]} onValueChange={([v]) => setEpsilon(v)} min={0.01} max={0.5} step={0.01} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Episodes: {numEpisodes}</label>
          <Slider value={[numEpisodes]} onValueChange={([v]) => setNumEpisodes(v)} min={50} max={1000} step={25} />
        </div>
        <button onClick={handleResample}
          className="rounded bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white text-sm px-3 py-2">
          Re-sample
        </button>
      </div>
      <CanvasChart data={pathTraces} layout={{
        title: { text: 'Learned greedy paths (4x12 grid, start=bottom-left, goal=bottom-right)' },
        xaxis: { title: { text: 'column' }, range: [-0.5, 11.5] },
        yaxis: { title: { text: 'row' }, range: [3.8, -0.5] },
        height: 260,
      }} style={{ width: '100%' }} />
      <div className="mt-4" />
      <CanvasChart data={rewardTraces} layout={{
        title: { text: 'Smoothed reward per episode (20-episode window)' },
        xaxis: { title: { text: 'episode' } },
        yaxis: { title: { text: 'reward' } },
        height: 320,
      }} style={{ width: '100%' }} />
    </div>
  );
}
