"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Repressilator: Three-gene oscillatory circuit (Elowitz & Leibler, 2000).
 *
 * dm_i/dt = alpha / (1 + p_j^n) + alpha0 - m_i
 * dp_i/dt = beta_t * (m_i - p_i)
 *
 * Repression chain: gene 1 <-- protein 3, gene 2 <-- protein 1, gene 3 <-- protein 2
 */

interface SimulationResult {
  time: number[];
  p1: number[];
  p2: number[];
  p3: number[];
  period: number | null;
}

function simulateRepressilator(
  n: number,
  alpha: number,
  betaT: number,
  alpha0: number,
): SimulationResult {
  const dt = 0.01;
  const T = 100;
  const totalSteps = Math.round(T / dt);

  // State: [m1, m2, m3, p1, p2, p3]
  let m1 = 1.0;
  let m2 = 1.5;
  let m3 = 2.0;
  let p1 = 1.0;
  let p2 = 1.5;
  let p3 = 2.0;

  // Subsample every 10th point for display
  const subsample = 10;
  const displayLen = Math.floor(totalSteps / subsample) + 1;
  const time: number[] = new Array(displayLen);
  const p1Arr: number[] = new Array(displayLen);
  const p2Arr: number[] = new Array(displayLen);
  const p3Arr: number[] = new Array(displayLen);

  let idx = 0;

  for (let step = 0; step <= totalSteps; step++) {
    if (step % subsample === 0) {
      time[idx] = step * dt;
      p1Arr[idx] = p1;
      p2Arr[idx] = p2;
      p3Arr[idx] = p3;
      idx++;
    }

    // Repression: gene 1 repressed by p3, gene 2 by p1, gene 3 by p2
    const dm1 = (alpha / (1 + Math.pow(p3, n)) + alpha0 - m1) * dt;
    const dm2 = (alpha / (1 + Math.pow(p1, n)) + alpha0 - m2) * dt;
    const dm3 = (alpha / (1 + Math.pow(p2, n)) + alpha0 - m3) * dt;

    const dp1 = betaT * (m1 - p1) * dt;
    const dp2 = betaT * (m2 - p2) * dt;
    const dp3 = betaT * (m3 - p3) * dt;

    m1 += dm1;
    m2 += dm2;
    m3 += dm3;
    p1 += dp1;
    p2 += dp2;
    p3 += dp3;

    // Clamp to avoid negative concentrations
    if (m1 < 0) m1 = 0;
    if (m2 < 0) m2 = 0;
    if (m3 < 0) m3 = 0;
    if (p1 < 0) p1 = 0;
    if (p2 < 0) p2 = 0;
    if (p3 < 0) p3 = 0;
  }

  // Estimate period from p1: find local maxima in the second half of the trace
  const halfIdx = Math.floor(displayLen / 2);
  const peaks: number[] = [];
  for (let i = halfIdx + 1; i < displayLen - 1; i++) {
    if (p1Arr[i] > p1Arr[i - 1] && p1Arr[i] > p1Arr[i + 1]) {
      // Only count peaks that are meaningfully above the trough
      const localMin = Math.min(
        ...p1Arr.slice(Math.max(halfIdx, i - 20), i),
      );
      if (p1Arr[i] - localMin > 0.1) {
        peaks.push(time[i]);
      }
    }
  }

  let period: number | null = null;
  if (peaks.length >= 2) {
    let totalInterval = 0;
    for (let i = 1; i < peaks.length; i++) {
      totalInterval += peaks[i] - peaks[i - 1];
    }
    period = totalInterval / (peaks.length - 1);
  }

  return { time, p1: p1Arr, p2: p2Arr, p3: p3Arr, period };
}

export default function Repressilator({}: SimulationComponentProps) {
  const [n, setN] = useState(3.0);
  const [alpha, setAlpha] = useState(10);
  const [betaT, setBetaT] = useState(1.0);
  const [alpha0, setAlpha0] = useState(0.01);

  const result = useMemo(
    () => simulateRepressilator(n, alpha, betaT, alpha0),
    [n, alpha, betaT, alpha0],
  );

  const oscillating = result.period !== null;

  const chartData = useMemo(
    () => [
      {
        x: result.time,
        y: result.p1,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#3b82f6', width: 2 },
        name: 'Protein A',
      },
      {
        x: result.time,
        y: result.p2,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#22c55e', width: 2 },
        name: 'Protein B',
      },
      {
        x: result.time,
        y: result.p3,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#f97316', width: 2 },
        name: 'Protein C',
      },
    ],
    [result],
  );

  const chartLayout = useMemo(() => {
    const allP = [...result.p1, ...result.p2, ...result.p3];
    const yMax = Math.max(...allP) * 1.1 || 12;
    return {
      height: 380,
      margin: { t: 20, b: 50, l: 55, r: 20 },
      xaxis: {
        title: { text: 'Time' },
        range: [0, 100],
      },
      yaxis: {
        title: { text: 'Protein concentration' },
        range: [0, yMax],
      },
      showlegend: true,
    };
  }, [result]);

  return (
    <SimulationPanel title="Repressilator: Three-Gene Oscillator">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Hill coefficient n: {n.toFixed(1)}
          </SimulationLabel>
          <Slider
            min={1}
            max={8}
            step={0.5}
            value={[n]}
            onValueChange={([v]) => setN(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>
            Repression strength &alpha;: {alpha}
          </SimulationLabel>
          <Slider
            min={1}
            max={50}
            step={1}
            value={[alpha]}
            onValueChange={([v]) => setAlpha(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>
            Protein-mRNA coupling &beta;<sub>t</sub>: {betaT.toFixed(1)}
          </SimulationLabel>
          <Slider
            min={0.1}
            max={5}
            step={0.1}
            value={[betaT]}
            onValueChange={([v]) => setBetaT(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>
            Basal rate &alpha;<sub>0</sub>: {alpha0.toFixed(2)}
          </SimulationLabel>
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={[alpha0]}
            onValueChange={([v]) => setAlpha0(v)}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      {/* Chart */}
      <SimulationMain>
      <CanvasChart
        data={chartData}
        layout={chartLayout}
        style={{ width: '100%' }}
      />
      </SimulationMain>

      <SimulationResults>
        <div className="text-sm font-medium text-[var(--text-strong)]">
          {oscillating
            ? `Period \u2248 ${result.period!.toFixed(1)} time units`
            : 'No oscillations detected'}
        </div>
      </SimulationResults>

      {/* What to notice */}
      <div className="mt-4 text-sm text-[var(--text-muted)]">
        <p>
          <strong className="text-[var(--text-muted)]">What to notice:</strong>{' '}
          {oscillating
            ? 'Clear oscillations! The three proteins take turns dominating, 120\u00B0 out of phase.'
            : 'Traces converge to a fixed point \u2014 cooperativity is too weak for oscillations.'}
        </p>
      </div>
    </SimulationPanel>
  );
}
