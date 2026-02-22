"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * ProductionDegradationCrossings: Plot production rate f(X) and degradation rate gamma*X
 * on the same axes. Steady states are where the curves cross.
 *
 * Production function (positive feedback with basal leak):
 *   f(X) = beta * (X/K)^n / (1 + (X/K)^n) + beta0
 *
 * Degradation: g(X) = gamma * X
 *
 * Crossings change from 1 (monostable) to 3 (bistable) as Hill coefficient
 * and production rate increase.
 */

const K = 1;
const X_MAX = 6;

function production(X: number, beta: number, n: number, beta0: number): number {
  if (X <= 0) return beta0;
  const xk = Math.pow(X / K, n);
  return beta * xk / (1 + xk) + beta0;
}

function findCrossings(
  beta: number, n: number, beta0: number, gamma: number, numPoints: number,
): Array<{ x: number; stable: boolean }> {
  const crossings: Array<{ x: number; stable: boolean }> = [];
  const dx = X_MAX / numPoints;

  for (let i = 1; i <= numPoints; i++) {
    const xPrev = (i - 1) * dx;
    const xCurr = i * dx;

    const diffPrev = production(xPrev, beta, n, beta0) - gamma * xPrev;
    const diffCurr = production(xCurr, beta, n, beta0) - gamma * xCurr;

    if (diffPrev * diffCurr < 0) {
      // Bisection to find precise crossing
      let lo = xPrev;
      let hi = xCurr;
      for (let j = 0; j < 40; j++) {
        const mid = (lo + hi) / 2;
        const diffMid = production(mid, beta, n, beta0) - gamma * mid;
        if (diffMid * (production(lo, beta, n, beta0) - gamma * lo) <= 0) {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      const xRoot = (lo + hi) / 2;

      // Stability: check sign of d(f-g)/dX at the root
      const eps = 1e-5;
      const dfdx = (production(xRoot + eps, beta, n, beta0) - production(xRoot - eps, beta, n, beta0)) / (2 * eps);
      const dgdx = gamma;
      const stable = dfdx - dgdx < 0;

      crossings.push({ x: xRoot, stable });
    }
  }

  return crossings;
}

export default function ProductionDegradationCrossings({}: SimulationComponentProps) {
  const [beta, setBeta] = useState(5.0);
  const [n, setN] = useState(4);
  const [beta0, setBeta0] = useState(0.1);
  const [gamma, setGamma] = useState(1.0);

  const { prodX, prodY, degY, crossings } = useMemo(() => {
    const numPoints = 500;
    const prodX: number[] = [];
    const prodY: number[] = [];
    const degY: number[] = [];

    for (let i = 0; i <= numPoints; i++) {
      const x = (i / numPoints) * X_MAX;
      prodX.push(x);
      prodY.push(production(x, beta, n, beta0));
      degY.push(gamma * x);
    }

    const crossings = findCrossings(beta, n, beta0, gamma, 2000);

    return { prodX, prodY, degY, crossings };
  }, [beta, n, beta0, gamma]);

  const yMax = Math.max(beta + beta0, gamma * X_MAX) * 1.1;

  const stableCrossings = crossings.filter(c => c.stable);
  const unstableCrossings = crossings.filter(c => !c.stable);

  const numCrossings = crossings.length;
  const isBistable = numCrossings === 3;

  const chartData: any[] = [
    // Production curve
    {
      x: prodX,
      y: prodY,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#3b82f6', width: 2.5 },
      name: 'Production f(X)',
    },
    // Degradation line
    {
      x: prodX,
      y: degY,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#ef4444', width: 2.5 },
      name: 'Degradation \u03B3X',
    },
  ];

  // Stable crossings
  if (stableCrossings.length > 0) {
    chartData.push({
      x: stableCrossings.map(c => c.x),
      y: stableCrossings.map(c => gamma * c.x),
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#22c55e', size: 12, line: { width: 2, color: '#ffffff' } },
      name: 'Stable steady state',
    });
  }

  // Unstable crossings
  if (unstableCrossings.length > 0) {
    chartData.push({
      x: unstableCrossings.map(c => c.x),
      y: unstableCrossings.map(c => gamma * c.x),
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#9ca3af', size: 10, line: { width: 2, color: '#ffffff' } },
      name: 'Unstable steady state',
    });
  }

  return (
    <SimulationPanel title="Production vs. Degradation: Finding Steady States">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Max production &beta;: {beta.toFixed(1)}
          </SimulationLabel>
          <Slider value={[beta]} onValueChange={([v]) => setBeta(v)} min={0.5} max={10} step={0.1} />
        </div>
        <div>
          <SimulationLabel>
            Hill coefficient n: {n}
          </SimulationLabel>
          <Slider value={[n]} onValueChange={([v]) => setN(v)} min={1} max={8} step={1} />
        </div>
        <div>
          <SimulationLabel>
            Basal rate &beta;&#8320;: {beta0.toFixed(2)}
          </SimulationLabel>
          <Slider value={[beta0]} onValueChange={([v]) => setBeta0(v)} min={0} max={1} step={0.01} />
        </div>
        <div>
          <SimulationLabel>
            Degradation rate &gamma;: {gamma.toFixed(2)}
          </SimulationLabel>
          <Slider value={[gamma]} onValueChange={([v]) => setGamma(v)} min={0.1} max={3.0} step={0.05} />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <CanvasChart
        data={chartData}
        layout={{
          height: 420,
          margin: { t: 20, b: 55, l: 55, r: 20 },
          xaxis: {
            title: { text: 'Protein concentration X' },
            range: [0, X_MAX],
          },
          yaxis: {
            title: { text: 'Rate' },
            range: [0, yMax],
          },
          showlegend: true,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>

      <SimulationResults>
        <div className="text-sm font-medium">
          <span className={isBistable ? 'text-[#f97316]' : 'text-[#22c55e]'}>
            {numCrossings} crossing{numCrossings !== 1 ? 's' : ''} &mdash;{' '}
            {isBistable ? 'Bistable (two stable + one unstable)' :
             numCrossings === 1 ? 'Monostable (one stable steady state)' :
             `${numCrossings} steady states`}
          </span>
        </div>
      </SimulationResults>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          The <span style={{ color: '#3b82f6' }}>blue curve</span> is the production rate
          f(X) &mdash; an S-shaped Hill function for positive feedback.
          The <span style={{ color: '#ef4444' }}>red line</span> is the degradation rate &gamma;X.
          Where they cross, production equals degradation &mdash; that is a steady state.
          With n = 1 (no cooperativity), there is always exactly one crossing.
          Increase n to 4 or higher and adjust &gamma; to find the parameter range where
          three crossings appear: two <span style={{ color: '#22c55e' }}>stable</span> (green)
          and one <span style={{ color: '#9ca3af' }}>unstable</span> (grey).
          That is bistability.
        </p>
      </div>
    </SimulationPanel>
  );
}
