"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function runBetHedging(
  p: number,
  noise: number,
  investPerRound: number,
  nsteps: number,
  winMultiplier: number,
  lossMultiplier: number,
  numAgents: number
): number[][] {
  const allCapitals: number[][] = [];

  for (let a = 0; a < numAgents; a++) {
    const capital: number[] = [1];
    for (let i = 0; i < nsteps; i++) {
      const probLoss = 1 / (2 * (1 - p)) + (Math.random() * 2 - 1) * noise;
      const invested = capital[i] * investPerRound;
      const kept = capital[i] - invested;
      if (Math.random() > probLoss) {
        capital.push(kept + invested * winMultiplier);
      } else {
        capital.push(kept + invested * lossMultiplier);
      }
    }
    allCapitals.push(capital);
  }

  return allCapitals;
}

export default function BetHedging({}: SimulationComponentProps) {
  const [p, setP] = useState(0.05);
  const [noise, setNoise] = useState(1.0);
  const [investPerRound, setInvestPerRound] = useState(0.5);
  const [nsteps, setNsteps] = useState(500);
  const [winMultiplier, setWinMultiplier] = useState(1.5);
  const [lossMultiplier, setLossMultiplier] = useState(0.5);
  const [seed, setSeed] = useState(0);

  const numAgents = 5;

  const results = useMemo(() => {
    return runBetHedging(p, noise, investPerRound, nsteps, winMultiplier, lossMultiplier, numAgents);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p, noise, investPerRound, nsteps, winMultiplier, lossMultiplier, seed]);

  const colors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444'];

  return (
    <SimulationPanel title="Bet Hedging: Capital Over Time">
      <SimulationSettings>
        <SimulationButton variant="primary" onClick={() => setSeed(s => s + 1)}>
          Re-run
        </SimulationButton>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <SimulationLabel>Fear probability p: {p.toFixed(2)}</SimulationLabel>
            <Slider
              min={0}
              max={0.3}
              step={0.01}
              value={[p]}
              onValueChange={([v]) => setP(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Noise: {noise.toFixed(1)}</SimulationLabel>
            <Slider
              min={0}
              max={3}
              step={0.1}
              value={[noise]}
              onValueChange={([v]) => setNoise(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Invest fraction: {investPerRound.toFixed(2)}</SimulationLabel>
            <Slider
              min={0.01}
              max={1}
              step={0.01}
              value={[investPerRound]}
              onValueChange={([v]) => setInvestPerRound(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Win multiplier: {winMultiplier.toFixed(1)}</SimulationLabel>
            <Slider
              min={1}
              max={4}
              step={0.1}
              value={[winMultiplier]}
              onValueChange={([v]) => setWinMultiplier(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Loss multiplier: {lossMultiplier.toFixed(2)}</SimulationLabel>
            <Slider
              min={0}
              max={1}
              step={0.05}
              value={[lossMultiplier]}
              onValueChange={([v]) => setLossMultiplier(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Steps: {nsteps}</SimulationLabel>
            <Slider
              min={50}
              max={3000}
              step={50}
              value={[nsteps]}
              onValueChange={([v]) => setNsteps(v)}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasChart
        data={results.map((capital, idx) => ({
          y: capital,
          type: 'scatter' as const,
          mode: 'lines' as const,
          line: { color: colors[idx % colors.length], width: 1.5 },
          name: `Agent ${idx + 1}`,
        }))}
        layout={{
          title: { text: 'Bet Hedging: Capital Over Time', font: { size: 14 } },
          xaxis: { title: { text: 'Timestep' } },
          yaxis: { title: { text: 'Capital' }, type: 'log' },
          showlegend: true,
          margin: { t: 40, r: 20, b: 50, l: 60 },
        }}
        style={{ width: '100%', height: 400 }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
