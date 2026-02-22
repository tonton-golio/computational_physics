"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Flow through a converging-diverging channel (Venturi tube).
 * Plots velocity and pressure along the channel centreline using
 * Bernoulli's equation and continuity (A * v = const).
 */

export default function BernoulliStreamline({}: SimulationComponentProps) {
  const [constriction, setConstriction] = useState(0.4); // min area / inlet area
  const [inletVelocity, setInletVelocity] = useState(2.0); // m/s

  const { velData, pressData, areaData } = useMemo(() => {
    const N = 200;
    const L = 1; // channel length normalised
    const rho = 1000; // water

    const xArr: number[] = [];
    const areaArr: number[] = [];
    const velArr: number[] = [];
    const pressArr: number[] = [];

    // Channel area profile: smooth constriction using a Gaussian bump
    // A(x) = A_in * (1 - (1 - r) * exp(-(x-0.5)^2 / 0.02))
    const Ain = 1;
    const r = constriction; // minimum area ratio

    for (let i = 0; i <= N; i++) {
      const x = (i / N) * L;
      xArr.push(x);
      const A = Ain * (1 - (1 - r) * Math.exp(-((x - 0.5) ** 2) / 0.02));
      areaArr.push(A);

      // Continuity: v(x) = v_in * A_in / A(x)
      const v = inletVelocity * Ain / A;
      velArr.push(v);

      // Bernoulli: p + 0.5*rho*v^2 = p_in + 0.5*rho*v_in^2
      // p(x) = p_in + 0.5*rho*(v_in^2 - v^2)
      // Use p_in = 101325 Pa (1 atm)
      const pIn = 101325;
      const p = pIn + 0.5 * rho * (inletVelocity ** 2 - v ** 2);
      pressArr.push(p / 1000); // kPa
    }

    return {
      velData: { x: xArr, y: velArr },
      pressData: { x: xArr, y: pressArr },
      areaData: { x: xArr, y: areaArr },
    };
  }, [constriction, inletVelocity]);

  const velTraces = useMemo(() => [
    {
      type: 'scatter' as const,
      mode: 'lines' as const,
      x: velData.x,
      y: velData.y,
      name: 'Velocity',
      line: { color: '#3b82f6', width: 2 },
    },
  ], [velData]);

  const pressTraces = useMemo(() => [
    {
      type: 'scatter' as const,
      mode: 'lines' as const,
      x: pressData.x,
      y: pressData.y,
      name: 'Pressure',
      line: { color: '#ef4444', width: 2 },
    },
  ], [pressData]);

  const areaTraces = useMemo(() => [
    {
      type: 'scatter' as const,
      mode: 'lines' as const,
      x: areaData.x,
      y: areaData.y,
      name: 'Area ratio',
      line: { color: '#10b981', width: 2 },
      fill: 'tozeroy' as const,
      fillcolor: 'rgba(16,185,129,0.08)',
    },
  ], [areaData]);

  return (
    <SimulationPanel title="Bernoulli's Equation in a Converging-Diverging Channel" caption="Continuity requires A\u00b7v = const along a streamtube. Where the channel narrows, velocity increases and pressure drops (Bernoulli). Adjust the constriction ratio and inlet velocity.">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Constriction ratio A<sub>min</sub>/A<sub>in</sub>: {constriction.toFixed(2)}
          </SimulationLabel>
          <Slider min={0.1} max={0.95} step={0.01} value={[constriction]}
            onValueChange={([v]) => setConstriction(v)} className="w-full" />
        </div>
        <div>
          <SimulationLabel>
            Inlet velocity: {inletVelocity.toFixed(1)} m/s
          </SimulationLabel>
          <Slider min={0.5} max={10} step={0.1} value={[inletVelocity]}
            onValueChange={([v]) => setInletVelocity(v)} className="w-full" />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="space-y-4">
          <CanvasChart
            data={areaTraces}
            layout={{ xaxis: { title: { text: 'Position x/L' } }, yaxis: { title: { text: 'A/A_in' } } }}
            style={{ width: '100%', height: 180 }}
          />
          <CanvasChart
            data={velTraces}
            layout={{ xaxis: { title: { text: 'Position x/L' } }, yaxis: { title: { text: 'Velocity (m/s)' } } }}
            style={{ width: '100%', height: 200 }}
          />
          <CanvasChart
            data={pressTraces}
            layout={{ xaxis: { title: { text: 'Position x/L' } }, yaxis: { title: { text: 'Pressure (kPa)' } } }}
            style={{ width: '100%', height: 200 }}
          />
        </div>
      </SimulationMain>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Top: channel cross-section area. Middle: velocity along the centreline (peaks at the throat).
        Bottom: pressure (drops at the throat). This is the Venturi effect.
      </p>
    </SimulationPanel>
  );
}
