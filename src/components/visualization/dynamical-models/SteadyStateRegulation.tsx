"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Steady-state concentration for positive and negative regulation.
 *
 * Positive regulation:
 *   dP/dt = P^H / (1 + P^H) - Gamma_P * P
 *   Steady states are intersections of y = P^H/(1+P^H) and y = Gamma_P * P
 *
 * Negative regulation:
 *   dP/dt = 1 / (1 + P^H) - Gamma_P * P
 *   Steady states are intersections of y = 1/(1+P^H) and y = Gamma_P * P
 */
export default function SteadyStateRegulation({}: SimulationComponentProps) {
  const [H, setH] = useState(1);
  const [gammaP, setGammaP] = useState(0.5);

  const { xVals, yPosHill, yNegHill, yLinear } = useMemo(() => {
    const n = 500;
    const xMax = 4.0;
    const xVals: number[] = [];
    const yPosHill: number[] = [];
    const yNegHill: number[] = [];
    const yLinear: number[] = [];

    for (let i = 0; i <= n; i++) {
      const x = (i / n) * xMax;
      xVals.push(x);
      const xH = Math.pow(x, H);
      yPosHill.push(xH / (1 + xH));
      yNegHill.push(1.0 / (1 + xH));
      yLinear.push(gammaP * x);
    }

    return { xVals, yPosHill, yNegHill, yLinear };
  }, [H, gammaP]);

  const commonLayout = {
    height: 380,
    margin: { t: 40, b: 50, l: 50, r: 20 },
    xaxis: {
      title: { text: 'Protein concentration P' },
      range: [0, 4],
    },
    yaxis: {
      title: { text: 'y' },
      range: [0, 1.05],
    },
  };

  return (
    <SimulationPanel title="Steady-State Concentration: Positive vs. Negative Regulation">
      <SimulationConfig>
        <div>
          <SimulationLabel>Hill coefficient H: {H}</SimulationLabel>
          <Slider value={[H]} onValueChange={([v]) => setH(v)} min={1} max={10} step={1} />
        </div>
        <div>
          <SimulationLabel>Degradation rate Gamma_P: {gammaP.toFixed(2)}</SimulationLabel>
          <Slider value={[gammaP]} onValueChange={([v]) => setGammaP(v)} min={0.05} max={1.0} step={0.05} />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <CanvasChart
            data={[
              {
                x: xVals, y: yPosHill, type: 'scatter', mode: 'lines',
                line: { color: '#3b82f6', width: 2.5 },
                name: 'P^H / (1+P^H)',
              },
              {
                x: xVals, y: yLinear, type: 'scatter', mode: 'lines',
                line: { color: '#f97316', width: 2, dash: 'dash' },
                name: 'Gamma_P * P',
              },
            ] as any}
            layout={{
              ...commonLayout,
              title: {
                text: 'Positive regulation',
              },
              legend: { x: 0.5, y: 0.3, bgcolor: 'rgba(0,0,0,0.3)' },
            }}
            style={{ width: '100%' }}
          />
        </div>
        <div>
          <CanvasChart
            data={[
              {
                x: xVals, y: yNegHill, type: 'scatter', mode: 'lines',
                line: { color: '#22c55e', width: 2.5 },
                name: '1 / (1+P^H)',
              },
              {
                x: xVals, y: yLinear, type: 'scatter', mode: 'lines',
                line: { color: '#f97316', width: 2, dash: 'dash' },
                name: 'Gamma_P * P',
              },
            ] as any}
            layout={{
              ...commonLayout,
              title: {
                text: 'Negative regulation',
              },
              legend: { x: 0.5, y: 0.95, bgcolor: 'rgba(0,0,0,0.3)' },
            }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
      </SimulationMain>

      <div className="mt-4 text-sm text-[var(--text-muted)]">
        <p>
          The <strong className="text-[var(--text-muted)]">steady-state concentrations</strong> are found at the intersections
          of the production curve and the degradation line. For <em>positive regulation</em>,
          when the Hill coefficient H is large enough, the sigmoid production curve can intersect
          the linear degradation line at multiple points, creating <strong className="text-[var(--text-muted)]">bistability</strong>.
          For <em>negative regulation</em>, there is always a unique steady state.
        </p>
      </div>
    </SimulationPanel>
  );
}
