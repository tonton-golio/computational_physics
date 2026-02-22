"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Bifurcation diagram for a positive-feedback gene-regulation model.
 *
 * Model: dX/dt = beta * (X/K)^n / (1 + (X/K)^n) + beta0 - gamma * X
 *
 * Steady states satisfy: beta * (X/K)^n / (1 + (X/K)^n) + beta0 = gamma * X
 * i.e. f(X) = beta * (X/K)^n / (1 + (X/K)^n) + beta0 - gamma * X = 0
 *
 * Stability: df/dX < 0 at root => stable, df/dX > 0 => unstable.
 */

const BETA = 5;
const K = 1;
const BETA0 = 0.1;
const X_MAX = 6;
const X_STEP = 0.005;
const GAMMA_MIN = 0.01;
const GAMMA_MAX = 3.0;
const GAMMA_STEP = 0.02;
const BISECTION_ITER = 40;

function f(X: number, n: number, gamma: number): number {
  if (X <= 0) return BETA0; // At X=0, Hill term is 0
  const xk = Math.pow(X / K, n);
  return BETA * xk / (1 + xk) + BETA0 - gamma * X;
}

function dfdx(X: number, n: number, gamma: number): number {
  // Numerical derivative via central difference
  const h = 1e-6;
  return (f(X + h, n, gamma) - f(X - h, n, gamma)) / (2 * h);
}

function bisect(
  xLo: number,
  xHi: number,
  n: number,
  gamma: number,
): number {
  let lo = xLo;
  let hi = xHi;
  for (let i = 0; i < BISECTION_ITER; i++) {
    const mid = (lo + hi) / 2;
    if (f(mid, n, gamma) * f(lo, n, gamma) <= 0) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return (lo + hi) / 2;
}

interface SteadyState {
  x: number;
  stable: boolean;
}

function findSteadyStates(n: number, gamma: number): SteadyState[] {
  const roots: SteadyState[] = [];
  const steps = Math.ceil(X_MAX / X_STEP);
  let prevX = 0;
  let prevF = f(0, n, gamma);

  for (let i = 1; i <= steps; i++) {
    const curX = i * X_STEP;
    const curF = f(curX, n, gamma);

    if (prevF * curF < 0) {
      const root = bisect(prevX, curX, n, gamma);
      const deriv = dfdx(root, n, gamma);
      roots.push({ x: root, stable: deriv < 0 });
    }

    prevX = curX;
    prevF = curF;
  }

  return roots;
}

export default function BifurcationDiagram({}: SimulationComponentProps) {
  const [n, setN] = useState(4);
  const [gammaSlider, setGammaSlider] = useState(1.0);

  const { stableBranchX, stableBranchY, unstableBranchX, unstableBranchY, saddleNodes, currentStates } =
    useMemo(() => {
      const stableBranchX: number[] = [];
      const stableBranchY: number[] = [];
      const unstableBranchX: number[] = [];
      const unstableBranchY: number[] = [];

      // Track transitions for saddle-node detection
      let prevCount = 0;
      const saddleNodes: { gamma: number; x: number }[] = [];

      const gammaSteps = Math.round((GAMMA_MAX - GAMMA_MIN) / GAMMA_STEP);

      for (let gi = 0; gi <= gammaSteps; gi++) {
        const gamma = GAMMA_MIN + gi * GAMMA_STEP;
        const roots = findSteadyStates(n, gamma);

        for (const root of roots) {
          if (root.stable) {
            stableBranchX.push(gamma);
            stableBranchY.push(root.x);
          } else {
            unstableBranchX.push(gamma);
            unstableBranchY.push(root.x);
          }
        }

        // Detect saddle-node bifurcations (number of roots changes)
        const curCount = roots.length;
        if (prevCount !== curCount && gi > 0) {
          // Bifurcation near this gamma value
          const bifGamma = gamma - GAMMA_STEP / 2;
          // Find the approximate X at the bifurcation
          const bifRoots = findSteadyStates(n, bifGamma);
          if (bifRoots.length > 0) {
            // Pick the root closest to where unstable and stable branches meet
            // For saddle-node, choose the root with smallest |df/dX|
            let bestRoot = bifRoots[0];
            let bestAbsDeriv = Math.abs(dfdx(bifRoots[0].x, n, bifGamma));
            for (const r of bifRoots) {
              const ad = Math.abs(dfdx(r.x, n, bifGamma));
              if (ad < bestAbsDeriv) {
                bestAbsDeriv = ad;
                bestRoot = r;
              }
            }
            saddleNodes.push({ gamma: bifGamma, x: bestRoot.x });
          }
        }
        prevCount = curCount;
      }

      // Current steady states at the slider value
      const currentStates = findSteadyStates(n, gammaSlider);

      return {
        stableBranchX,
        stableBranchY,
        unstableBranchX,
        unstableBranchY,
        saddleNodes,
        currentStates,
      };
    }, [n, gammaSlider]);

  const chartData: any[] = [
    // Stable branch (scatter points to avoid connecting separate branches)
    {
      x: stableBranchX,
      y: stableBranchY,
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#3b82f6', size: 3 },
      name: 'Stable',
    },
    // Unstable branch
    {
      x: unstableBranchX,
      y: unstableBranchY,
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#9ca3af', size: 2 },
      name: 'Unstable',
    },
    // Vertical marker line at current gamma
    {
      x: [gammaSlider, gammaSlider],
      y: [0, X_MAX],
      type: 'scatter',
      mode: 'lines',
      line: { color: '#f97316', width: 1.5, dash: 'dash' },
      showlegend: false,
    },
    // Current steady states as filled circles
    {
      x: currentStates.map(() => gammaSlider),
      y: currentStates.map((s) => s.x),
      type: 'scatter',
      mode: 'markers',
      marker: {
        color: currentStates.map((s) => (s.stable ? '#3b82f6' : '#9ca3af')),
        size: 10,
        line: { width: 2, color: '#f97316' },
      },
      name: 'Current states',
    },
  ];

  // Mark saddle-node bifurcation points
  if (saddleNodes.length > 0) {
    chartData.push({
      x: saddleNodes.map((sn) => sn.gamma),
      y: saddleNodes.map((sn) => sn.x),
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#ef4444', size: 8, symbol: 'diamond' },
      name: 'Saddle-node',
    });
  }

  const chartLayout = {
    height: 420,
    margin: { t: 40, b: 50, l: 55, r: 20 },
    xaxis: {
      title: { text: 'Degradation rate \u03B3' },
      range: [0, GAMMA_MAX],
    },
    yaxis: {
      title: { text: 'Steady-state X' },
      range: [0, X_MAX],
    },
  };

  return (
    <SimulationPanel title="Bifurcation Diagram: Positive Feedback">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Hill coefficient n: {n}
          </SimulationLabel>
          <Slider
            value={[n]}
            onValueChange={([v]) => setN(v)}
            min={1}
            max={8}
            step={1}
          />
        </div>
        <div>
          <SimulationLabel>
            Degradation rate {'\u03B3'}: {gammaSlider.toFixed(2)}
          </SimulationLabel>
          <Slider
            value={[gammaSlider]}
            onValueChange={([v]) => setGammaSlider(v)}
            min={0.1}
            max={3.0}
            step={0.05}
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <CanvasChart
        data={chartData}
        layout={chartLayout}
        style={{ width: '100%' }}
      />
      </SimulationMain>

      <SimulationResults>
      {saddleNodes.length > 0 && (
        <div className="text-sm text-[var(--text-muted)]">
          <strong className="text-[var(--text-muted)]">Saddle-node bifurcations</strong> at{' '}
          {saddleNodes
            .map((sn) => `\u03B3 \u2248 ${sn.gamma.toFixed(2)}`)
            .join(', ')}
        </div>
      )}
      </SimulationResults>

      <div className="mt-4 text-sm text-[var(--text-muted)]">
        <p>
          Sweep {'\u03B3'} from low to high: the system stays on the upper branch until the
          saddle-node, then jumps down. Sweep back: it stays on the lower branch until the other
          saddle-node, then jumps up. The system remembers which branch it was on — this is{' '}
          <strong className="text-[var(--text-muted)]">hysteresis</strong>.
        </p>
      </div>
    </SimulationPanel>
  );
}
