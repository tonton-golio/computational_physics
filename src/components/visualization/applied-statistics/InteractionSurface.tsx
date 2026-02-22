"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function InteractionSurface({}: SimulationComponentProps) {
  const [mainA, setMainA] = useState(2.0);
  const [mainB, setMainB] = useState(1.5);
  const [interaction, setInteraction] = useState(0.0);

  const { traces } = useMemo(() => {
    // 2-factor experiment: A has 3 levels, B has 3 levels
    const aLevels = [0, 1, 2];
    const bLevels = [0, 1, 2];
    const bLabels = ['B=Low', 'B=Med', 'B=High'];
    const colors = ['#3b82f6', '#10b981', '#f59e0b'];

    // Response: y = baseline + mainA*a + mainB*b + interaction*a*b
    const baseline = 10;
    const result: any[] = [];

    for (let bi = 0; bi < bLevels.length; bi++) {
      const b = bLevels[bi];
      const xs: number[] = [];
      const ys: number[] = [];
      for (const a of aLevels) {
        xs.push(a);
        ys.push(baseline + mainA * a + mainB * b + interaction * a * b);
      }
      result.push({
        x: xs, y: ys, type: 'scatter', mode: 'lines+markers',
        line: { color: colors[bi], width: 2.5 },
        marker: { color: colors[bi], size: 8 },
        name: bLabels[bi],
      });
    }

    return { traces: result };
  }, [mainA, mainB, interaction]);

  return (
    <SimulationPanel title="Factorial Experiment: Interaction Effects">
      <SimulationConfig>
        <div className="grid grid-cols-3 gap-6">
          <div>
            <SimulationLabel>Main effect A: {mainA.toFixed(1)}</SimulationLabel>
            <Slider value={[mainA]} onValueChange={([v]) => setMainA(v)} min={-3} max={5} step={0.1} />
          </div>
          <div>
            <SimulationLabel>Main effect B: {mainB.toFixed(1)}</SimulationLabel>
            <Slider value={[mainB]} onValueChange={([v]) => setMainB(v)} min={-3} max={5} step={0.1} />
          </div>
          <div>
            <SimulationLabel>Interaction A*B: {interaction.toFixed(1)}</SimulationLabel>
            <Slider value={[interaction]} onValueChange={([v]) => setInteraction(v)} min={-3} max={3} step={0.1} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={traces}
        layout={{
          height: 400,
          xaxis: { title: { text: 'Factor A level' }, tickvals: [0, 1, 2], ticktext: ['Low', 'Med', 'High'] },
          yaxis: { title: { text: 'Response' } },
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          {Math.abs(interaction) < 0.3
            ? 'No interaction: lines are parallel. The effect of A does not depend on B.'
            : 'Interaction present: lines are not parallel. The effect of A depends on the level of B.'}
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
