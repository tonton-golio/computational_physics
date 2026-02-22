"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { CanvasChart, type ChartTrace, type ChartLayout } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Two-Mode Squeezed Vacuum Entanglement
 *
 * Visualises entanglement properties of the TMSV state as a function
 * of the squeeze parameter r.
 *
 * Duan criterion: D(r) = e^{-2r}  (entangled when D < 1)
 * Thermal photons per mode: n_bar = sinh^2(r)
 * Logarithmic negativity: E_N = 2r / ln(2)
 */

function computeCurves(rMax: number, nPoints: number) {
  const rs: number[] = [];
  const duanVals: number[] = [];
  const nbarVals: number[] = [];
  const enVals: number[] = [];

  for (let i = 0; i <= nPoints; i++) {
    const r = (rMax * i) / nPoints;
    rs.push(r);
    duanVals.push(Math.exp(-2 * r));
    nbarVals.push(Math.sinh(r) ** 2);
    enVals.push((2 * r) / Math.LN2);
  }

  return { rs, duanVals, nbarVals, enVals };
}

export default function TMSVEntanglement({}: SimulationComponentProps) {
  const [squeezeR, setSqueezeR] = useState(1.0);

  const rMax = 3;
  const nPoints = 300;

  const { rs, duanVals, nbarVals, enVals } = useMemo(
    () => computeCurves(rMax, nPoints),
    [],
  );

  const currentD = Math.exp(-2 * squeezeR);
  const currentNbar = Math.sinh(squeezeR) ** 2;
  const currentEN = (2 * squeezeR) / Math.LN2;

  // --- Duan chart ---
  const duanTraces: ChartTrace[] = [
    {
      x: rs,
      y: duanVals,
      type: 'scatter',
      mode: 'lines',
      line: { color: 'var(--accent)', width: 2 },
      name: 'D(r)',
    },
    {
      x: [0, rMax],
      y: [1, 1],
      type: 'scatter',
      mode: 'lines',
      line: { color: 'var(--text-muted)', width: 1, dash: 'dash' },
      name: 'D = 1 threshold',
    },
    {
      x: [squeezeR],
      y: [currentD],
      type: 'scatter',
      mode: 'markers',
      marker: { color: 'var(--accent)', size: 8 },
      name: 'Current r',
    },
  ];

  const duanLayout: ChartLayout = {
    title: { text: 'Duan Criterion D(r)' },
    xaxis: { title: { text: 'Squeeze parameter r' } },
    yaxis: { title: { text: 'D(r)' }, range: [0, 1.15] },
    height: 320,
    margin: { t: 35, r: 15, b: 45, l: 50 },
    showlegend: false,
  };

  // --- n_bar and E_N chart ---
  const entTraces: ChartTrace[] = [
    {
      x: rs,
      y: nbarVals,
      type: 'scatter',
      mode: 'lines',
      line: { color: 'var(--accent)', width: 2 },
      name: 'n_bar(r)',
    },
    {
      x: rs,
      y: enVals,
      type: 'scatter',
      mode: 'lines',
      line: { color: 'var(--accent-secondary, #e67e22)', width: 2 },
      name: 'E_N(r)',
    },
    {
      x: [squeezeR],
      y: [currentNbar],
      type: 'scatter',
      mode: 'markers',
      marker: { color: 'var(--accent)', size: 8 },
      name: 'Current n_bar',
    },
    {
      x: [squeezeR],
      y: [currentEN],
      type: 'scatter',
      mode: 'markers',
      marker: { color: 'var(--accent-secondary, #e67e22)', size: 8 },
      name: 'Current E_N',
    },
  ];

  const entLayout: ChartLayout = {
    title: { text: 'Thermal Photons & Entanglement' },
    xaxis: { title: { text: 'Squeeze parameter r' } },
    yaxis: { title: { text: 'Value' } },
    height: 320,
    margin: { t: 35, r: 15, b: 45, l: 50 },
    showlegend: true,
  };

  return (
    <SimulationPanel title="Two-Mode Squeezed Vacuum Entanglement" caption="Entanglement properties of the TMSV state. The Duan criterion D(r) drops below 1 for any r > 0, confirming entanglement. Each mode individually looks thermal with mean photon number sinh²(r).">
      <SimulationConfig>
        <div>
          <SimulationLabel>Squeeze parameter r: {squeezeR.toFixed(2)}</SimulationLabel>
          <Slider
            value={[squeezeR]}
            onValueChange={(v) => setSqueezeR(v[0])}
            min={0}
            max={rMax}
            step={0.01}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasChart
            data={duanTraces}
            layout={duanLayout}
            style={{ width: '100%', height: '320px' }}
          />
          <CanvasChart
            data={entTraces}
            layout={entLayout}
            style={{ width: '100%', height: '320px' }}
          />
        </div>
      </SimulationMain>

      <SimulationResults>
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Duan D(r)</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {currentD.toFixed(4)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Thermal n_bar</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {currentNbar.toFixed(2)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Log. neg. E_N</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {currentEN.toFixed(2)}
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
