"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { CanvasChart, type ChartTrace, type ChartLayout } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Jaynes-Cummings Collapse & Revival
 *
 * Computes <sigma_z(t)> for an atom initially in |e> interacting with
 * a coherent field |alpha> in the Jaynes-Cummings model.
 *
 * <sigma_z(t)> = sum_n P(n) cos(Omega_n * t)
 *
 * where P(n) = e^{-|alpha|^2} |alpha|^{2n} / n! (Poisson)
 * and Omega_n = 2g * sqrt(n+1) (at resonance) or sqrt(Delta^2 + 4g^2(n+1)) (detuned)
 */
function factorial(n: number): number {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

export default function JaynesCummingsRevival({}: SimulationComponentProps) {
  const [meanN, setMeanN] = useState(15);
  const [coupling, setCoupling] = useState(1.0);
  const [detuning, setDetuning] = useState(0);

  const { times, sigmaZ, collapseTime, revivalTime } = useMemo(() => {
    const g = coupling;
    const Delta = detuning;
    const nbar = meanN;

    // Truncate Poisson sum at nbar + 5*sqrt(nbar)
    const nMax = Math.min(Math.ceil(nbar + 5 * Math.sqrt(nbar + 1) + 10), 150);

    // Poisson weights
    const P: number[] = [];
    for (let n = 0; n <= nMax; n++) {
      P.push(Math.exp(-nbar) * Math.pow(nbar, n) / factorial(n));
    }

    // Time range: show at least 2 revival times
    const tRevival = nbar > 0.5 ? 2 * Math.PI * Math.sqrt(nbar) / g : 20 / g;
    const tMax = Math.max(2.5 * tRevival, 30 / g);
    const nPoints = 800;
    const dt = tMax / nPoints;

    const times: number[] = [];
    const sigmaZ: number[] = [];

    for (let i = 0; i <= nPoints; i++) {
      const t = i * dt;
      times.push(t);

      let sz = 0;
      for (let n = 0; n <= nMax; n++) {
        const OmegaN = Math.sqrt(Delta * Delta + 4 * g * g * (n + 1));
        sz += P[n] * Math.cos(OmegaN * t);
      }
      sigmaZ.push(sz);
    }

    const collapseTime = nbar > 0.5 ? 1 / (g * Math.sqrt(nbar)) : Infinity;
    const revivalTime = nbar > 0.5 ? 2 * Math.PI * Math.sqrt(nbar) / g : Infinity;

    return { times, sigmaZ, collapseTime, revivalTime };
  }, [meanN, coupling, detuning]);

  const trace: ChartTrace = {
    x: times,
    y: sigmaZ,
    type: 'scatter',
    mode: 'lines',
    line: { color: 'var(--accent)', width: 1.5 },
    name: '<sigma_z(t)>',
  };

  const layout: ChartLayout = {
    title: { text: 'Jaynes-Cummings: Collapse & Revival' },
    xaxis: { title: { text: 'gt (dimensionless time)' } },
    yaxis: { title: { text: '<sigma_z>' }, range: [-1.1, 1.1] },
    height: 400,
    margin: { t: 35, r: 15, b: 45, l: 50 },
    showlegend: false,
  };

  return (
    <SimulationPanel title="Jaynes-Cummings Collapse & Revival" caption="Atomic inversion <sigma_z(t)> for an excited atom in a coherent cavity field. Different Fock components oscillate at incommensurable frequencies, causing collapse and then periodic revivals.">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <SimulationLabel>Mean photon number |alpha|^2: {meanN}</SimulationLabel>
            <Slider
              value={[meanN]}
              onValueChange={(v) => setMeanN(Math.round(v[0]))}
              min={1}
              max={50}
              step={1}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Coupling g: {coupling.toFixed(2)}</SimulationLabel>
            <Slider
              value={[coupling]}
              onValueChange={(v) => setCoupling(v[0])}
              min={0.1}
              max={3}
              step={0.05}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Detuning Delta/g: {(detuning / coupling).toFixed(1)}</SimulationLabel>
            <Slider
              value={[detuning]}
              onValueChange={(v) => setDetuning(v[0])}
              min={0}
              max={5}
              step={0.1}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasChart
          data={[trace]}
          layout={layout}
          style={{ width: '100%', height: '400px' }}
        />
      </SimulationMain>

      <SimulationResults>
        {/* Timescale readout */}
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Collapse time (gt_c)</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {isFinite(collapseTime) ? (collapseTime * coupling).toFixed(2) : '-'}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Revival time (gt_r)</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {isFinite(revivalTime) ? (revivalTime * coupling).toFixed(1) : '-'}
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
