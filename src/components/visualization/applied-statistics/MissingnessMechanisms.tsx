"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationResults, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function MissingnessMechanisms({}: SimulationComponentProps) {
  const [mechanism, setMechanism] = useState<'MCAR' | 'MAR' | 'MNAR'>('MCAR');

  const { obsX, obsY, missX, missY } = useMemo(() => {
    const N = 150;
    const oX: number[] = [];
    const oY: number[] = [];
    const mX: number[] = [];
    const mY: number[] = [];

    for (let i = 0; i < N; i++) {
      const x = Math.sin(i * 0.83 + 0.3) * 4 + 5;
      const y = 0.8 * x + Math.sin(i * 1.47 + 2.1) * 2 + 3;
      let isMissing = false;

      if (mechanism === 'MCAR') {
        // Random 30% missing, independent of data
        isMissing = ((i * 7 + 3) % 10) < 3;
      } else if (mechanism === 'MAR') {
        // Missing depends on x: high x values more likely to be missing
        isMissing = x > 7 && ((i * 3 + 1) % 5) < 3;
      } else {
        // MNAR: missing depends on y itself: high y values are missing
        isMissing = y > 9 && ((i * 3 + 1) % 4) < 3;
      }

      if (isMissing) {
        mX.push(x);
        mY.push(y);
      } else {
        oX.push(x);
        oY.push(y);
      }
    }

    return { obsX: oX, obsY: oY, missX: mX, missY: mY };
  }, [mechanism]);

  const descriptions: Record<string, string> = {
    MCAR: 'Missing Completely At Random: missingness is unrelated to the data. The observed data are a random subset.',
    MAR: 'Missing At Random: missingness depends on observed variables (high x). Analysis using observed data can still be valid.',
    MNAR: 'Missing Not At Random: missingness depends on the unobserved value itself (high y). This biases results.',
  };

  return (
    <SimulationPanel title="Missingness Mechanisms">
      <SimulationSettings>
        <SimulationToggle
          options={[
            { label: 'MCAR', value: 'MCAR' },
            { label: 'MAR', value: 'MAR' },
            { label: 'MNAR', value: 'MNAR' },
          ]}
          value={mechanism}
          onChange={(v) => setMechanism(v as 'MCAR' | 'MAR' | 'MNAR')}
        />
      </SimulationSettings>
      <SimulationMain>
      <CanvasChart
        data={[
          {
            x: obsX, y: obsY, type: 'scatter', mode: 'markers',
            marker: { color: '#3b82f6', size: 5 }, name: 'Observed',
          },
          {
            x: missX, y: missY, type: 'scatter', mode: 'markers',
            marker: { color: '#ef4444', size: 5, opacity: 0.5 }, name: 'Missing (shown for illustration)',
          },
        ]}
        layout={{
          height: 400,
          xaxis: { title: { text: 'x (observed variable)' } },
          yaxis: { title: { text: 'y (outcome)' } },
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          {descriptions[mechanism]} | Observed: {obsX.length} | Missing: {missX.length}
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
