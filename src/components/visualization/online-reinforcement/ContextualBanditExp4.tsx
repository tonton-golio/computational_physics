"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function ContextualBanditExp4({}: SimulationComponentProps) {
  const [rounds, setRounds] = useState(400);
  const contexts = useMemo(() => Array.from({ length: rounds }, (_, t) => (Math.sin(t / 18) > 0 ? 1 : 0)), [rounds]);
  const oracle = useMemo(() => contexts.map((c) => (c === 1 ? 1 : 0)), [contexts]);
  const exp4Choice = useMemo(
    () =>
      contexts.map((c, t) => {
        const pseudo = Math.sin((t + 1) * 91.177) * 43758.5453;
        const u = pseudo - Math.floor(pseudo);
        const exploreProb = Math.max(0.08, 0.5 * Math.exp(-t / 150));
        return u < exploreProb ? 1 - c : c;
      }),
    [contexts]
  );
  const rewardOracle = useMemo(() => oracle.map(() => 1), [oracle]);
  const rewardExp4 = useMemo(() => exp4Choice.map((a, t) => (a === oracle[t] ? 1 : 0)), [exp4Choice, oracle]);
  const cumOracle = useMemo(() => rewardOracle.map((_, i) => i + 1), [rewardOracle]);
  const cumExp4 = useMemo(() => rewardExp4.reduce<number[]>((acc, v, i) => { acc.push((acc[i - 1] ?? 0) + v); return acc; }, []), [rewardExp4]);
  const regret = useMemo(() => cumOracle.map((v, i) => v - cumExp4[i]), [cumOracle, cumExp4]);
  const x = Array.from({ length: rounds }, (_, i) => i + 1);

  return (
    <SimulationPanel title="Contextual Bandit (EXP4-style)">
      <SimulationConfig>
        <div>
          <SimulationLabel>Rounds: {rounds}</SimulationLabel>
          <Slider value={[rounds]} onValueChange={([v]) => setRounds(v)} min={100} max={1000} step={25} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          { x, y: cumExp4, type: 'scatter', mode: 'lines', name: 'Learner cumulative reward', line: { color: '#60a5fa', width: 2 } },
          { x, y: cumOracle, type: 'scatter', mode: 'lines', name: 'Best contextual policy', line: { color: '#4ade80', width: 2 } },
          { x, y: regret, type: 'scatter', mode: 'lines', name: 'Regret', line: { color: '#f87171', width: 2, dash: 'dot' } },
        ]}
        layout={{
          title: { text: 'Context-dependent actions reduce regret' },
          xaxis: { title: { text: 'round' } },
          yaxis: { title: { text: 'cumulative reward' } },
          height: 420,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
