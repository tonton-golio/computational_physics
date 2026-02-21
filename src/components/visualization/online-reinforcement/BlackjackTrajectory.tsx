'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Episode = { playerSum: number; dealerCard: number; hit: boolean; reward: number };

function playEpisode(rng: () => number): Episode[] {
  const dealerCard = Math.floor(rng() * 10) + 2;
  let playerSum = Math.floor(rng() * 11) + 12;
  const steps: Episode[] = [];
  while (playerSum < 20) {
    const hit = playerSum < 17 || rng() < 0.3;
    if (!hit) break;
    const card = Math.min(Math.floor(rng() * 10) + 2, 11);
    playerSum += card;
    steps.push({ playerSum: Math.min(playerSum, 21), dealerCard, hit, reward: 0 });
    if (playerSum > 21) { steps[steps.length - 1].reward = -1; return steps; }
  }
  let dealerSum = dealerCard + Math.floor(rng() * 10) + 2;
  while (dealerSum < 17) dealerSum += Math.min(Math.floor(rng() * 10) + 2, 11);
  const reward = dealerSum > 21 || playerSum > dealerSum ? 1 : playerSum === dealerSum ? 0 : -1;
  steps.push({ playerSum, dealerCard, hit: false, reward });
  return steps;
}

export default function BlackjackTrajectory({ id }: SimulationComponentProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [episodes, setEpisodes] = useState(200);
  const [seed, setSeed] = useState(42);

  const { valueChart, convergenceData } = useMemo(() => {
    const rng = mulberry32(seed);
    const counts: Record<string, number> = {};
    const sums: Record<string, number> = {};
    const snapshots: number[] = [];

    for (let ep = 0; ep < episodes; ep++) {
      const steps = playEpisode(rng);
      if (steps.length === 0) continue;
      const reward = steps[steps.length - 1].reward;
      const visited = new Set<string>();
      for (const s of steps) {
        const key = `${s.playerSum},${s.dealerCard}`;
        if (visited.has(key)) continue;
        visited.add(key);
        counts[key] = (counts[key] || 0) + 1;
        sums[key] = (sums[key] || 0) + reward;
      }
      const totalStates = Object.keys(counts).length;
      const avgAbs = totalStates > 0
        ? Object.keys(sums).reduce((a, k) => a + Math.abs(sums[k] / counts[k]), 0) / totalStates
        : 0;
      snapshots.push(avgAbs);
    }

    // Build bar chart: average V for each player sum (12..21), marginalised over dealer card
    const playerSums = Array.from({ length: 10 }, (_, i) => i + 12);
    const avgValues = playerSums.map((ps) => {
      let total = 0, cnt = 0;
      for (let dc = 2; dc <= 11; dc++) {
        const key = `${ps},${dc}`;
        if (counts[key]) { total += sums[key] / counts[key]; cnt++; }
      }
      return cnt > 0 ? total / cnt : 0;
    });

    const valueChart: ChartTrace[] = [{
      x: playerSums, y: avgValues, type: 'bar',
      marker: { color: '#3b82f6' }, name: 'V(player sum)',
    }];

    const convX = Array.from({ length: snapshots.length }, (_, i) => i + 1);
    const convergenceData: ChartTrace[] = [{
      x: convX, y: snapshots, type: 'scatter', mode: 'lines',
      name: 'Mean |V| across states', line: { color: '#10b981', width: 2 },
    }];

    return { valueChart, convergenceData };
  }, [episodes, seed]);

  const handleResample = useCallback(() => setSeed((s) => s + 1), []);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-[var(--text-strong)]">
        Blackjack Monte Carlo: Episode-Based Value Estimation
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        First-visit MC averages returns for each state over complete episodes.
        As episodes accumulate, value estimates converge from noise to structure.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Episodes: {episodes}</label>
          <Slider value={[episodes]} onValueChange={([v]) => setEpisodes(v)} min={20} max={2000} step={20} />
        </div>
        <button onClick={handleResample}
          className="rounded bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white text-sm px-3 py-2">
          Re-sample
        </button>
      </div>
      <CanvasChart data={valueChart} layout={{
        title: { text: 'Average state value by player sum (marginalized over dealer card)' },
        xaxis: { title: { text: 'player sum' } },
        yaxis: { title: { text: 'V(s)' } },
        height: 340,
      }} style={{ width: '100%' }} />
      <div className="mt-4" />
      <CanvasChart data={convergenceData} layout={{
        title: { text: 'Value estimate magnitude as episodes grow' },
        xaxis: { title: { text: 'episode' } },
        yaxis: { title: { text: 'mean |V|' } },
        height: 280,
      }} style={{ width: '100%' }} />
    </div>
  );
}
