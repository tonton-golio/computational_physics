"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { CanvasChart, type ChartTrace, type ChartLayout } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Vacuum Rabi Oscillation
 *
 * Plots P_e(t) and P_g(t) for a two-level atom in a single-mode cavity
 * using the generalised Rabi formula with detuning.
 *
 * Generalised Rabi frequency: Omega = sqrt(Delta^2 + 4g^2)
 * Inversion: <sigma_z(t)> = (Delta/Omega)^2 + (2g/Omega)^2 cos(Omega t)
 * P_e = (1 + <sigma_z>) / 2,  P_g = (1 - <sigma_z>) / 2
 */

export default function VacuumRabiOscillation({}: SimulationComponentProps) {
  const [coupling, setCoupling] = useState(0.05);
  const [detuning, setDetuning] = useState(0);

  const { times, Pe, Pg, omega, period } = useMemo(() => {
    const g = coupling;
    const Delta = detuning;
    const omega = Math.sqrt(Delta * Delta + 4 * g * g);
    const period = omega > 0 ? (2 * Math.PI) / omega : Infinity;

    const nPeriods = 10;
    const tMax = omega > 0 ? nPeriods * period : 200;
    const nPoints = 600;
    const dt = tMax / nPoints;

    const cosCoeff = (2 * g / omega) ** 2;
    const dcOffset = (Delta / omega) ** 2;

    const times: number[] = [];
    const Pe: number[] = [];
    const Pg: number[] = [];

    for (let i = 0; i <= nPoints; i++) {
      const t = i * dt;
      times.push(t);
      const sigmaZ = dcOffset + cosCoeff * Math.cos(omega * t);
      Pe.push((1 + sigmaZ) / 2);
      Pg.push((1 - sigmaZ) / 2);
    }

    return { times, Pe, Pg, omega, period };
  }, [coupling, detuning]);

  const traces: ChartTrace[] = [
    {
      x: times,
      y: Pe,
      type: 'scatter',
      mode: 'lines',
      line: { color: 'var(--accent)', width: 1.5 },
      name: 'P_e(t)',
    },
    {
      x: times,
      y: Pg,
      type: 'scatter',
      mode: 'lines',
      line: { color: 'var(--accent-secondary, #e67e22)', width: 1.5 },
      name: 'P_g(t)',
    },
  ];

  const layout: ChartLayout = {
    title: { text: 'Vacuum Rabi Oscillation' },
    xaxis: { title: { text: 'Time t' } },
    yaxis: { title: { text: 'Probability' }, range: [-0.05, 1.1] },
    height: 400,
    margin: { t: 35, r: 15, b: 45, l: 50 },
    showlegend: true,
  };

  return (
    <SimulationPanel title="Vacuum Rabi Oscillation" caption="Excited- and ground-state populations for a two-level atom coupled to a single cavity photon. Detuning reduces the oscillation amplitude and increases the frequency.">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <SimulationLabel>Coupling g: {coupling.toFixed(3)}</SimulationLabel>
            <Slider
              value={[coupling]}
              onValueChange={(v) => setCoupling(v[0])}
              min={0.01}
              max={0.2}
              step={0.005}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Detuning Delta: {detuning.toFixed(2)}</SimulationLabel>
            <Slider
              value={[detuning]}
              onValueChange={(v) => setDetuning(v[0])}
              min={-0.5}
              max={0.5}
              step={0.01}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasChart
          data={traces}
          layout={layout}
          style={{ width: '100%', height: '400px' }}
        />
      </SimulationMain>

      <SimulationResults>
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Rabi freq. Omega</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {omega.toFixed(4)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Period T</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {isFinite(period) ? period.toFixed(1) : '-'}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Vacuum Rabi splitting 2g</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {(2 * coupling).toFixed(4)}
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
