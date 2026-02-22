"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const FUNCTIONS = [
  {
    label: 'sin(x) at x = 1',
    f: (x: number) => Math.sin(x),
    fprime: Math.cos(1),
    fpp: Math.abs(Math.sin(1)),
    name: 'sin(x)',
  },
  {
    label: 'exp(x) at x = 0',
    f: (x: number) => Math.exp(x),
    fprime: 1,
    fpp: 1,
    name: 'exp(x)',
  },
  {
    label: 'x^3 at x = 1',
    f: (x: number) => x * x * x,
    fprime: 3,
    fpp: 6,
    name: 'x^3',
  },
];

export default function ErrorVsH({}: SimulationComponentProps) {
  const [funcIdx, setFuncIdx] = useState(0);
  const [epsExponent, setEpsExponent] = useState(-16);

  const machineEps = Math.pow(10, epsExponent);
  const chosen = FUNCTIONS[funcIdx];

  const data = useMemo(() => {
    const { f, fprime } = chosen;
    const x0 = funcIdx === 1 ? 0 : 1;
    const trueDeriv = fprime;

    const hValues: number[] = [];
    const forwardErrors: number[] = [];
    const centeredErrors: number[] = [];
    const richardsonErrors: number[] = [];
    const hTheoryO1: number[] = [];
    const hTheoryO2: number[] = [];
    const hTheoryO4: number[] = [];

    for (let e = -1; e >= -15; e -= 0.25) {
      const h = Math.pow(10, e);
      hValues.push(h);

      // Forward difference: (f(x+h) - f(x)) / h
      const fwd = (f(x0 + h) - f(x0)) / h;
      const fwdErr = Math.abs(fwd - trueDeriv);
      forwardErrors.push(Math.max(fwdErr, 1e-20));

      // Centered difference: (f(x+h) - f(x-h)) / (2h)
      const ctr = (f(x0 + h) - f(x0 - h)) / (2 * h);
      const ctrErr = Math.abs(ctr - trueDeriv);
      centeredErrors.push(Math.max(ctrErr, 1e-20));

      // Richardson extrapolation: (4*D(h/2) - D(h)) / 3 where D is centered diff
      const ctrHalf = (f(x0 + h / 2) - f(x0 - h / 2)) / h;
      const rich = (4 * ctrHalf - ctr) / 3;
      const richErr = Math.abs(rich - trueDeriv);
      richardsonErrors.push(Math.max(richErr, 1e-20));

      // Theoretical slopes with roundoff
      const M = chosen.fpp || 1;
      hTheoryO1.push(Math.max(M * h / 2 + 2 * machineEps / h, 1e-20));
      hTheoryO2.push(Math.max(M * h * h / 6 + 2 * machineEps / h, 1e-20));
      hTheoryO4.push(Math.max(M * h * h * h * h / 120 + 2 * machineEps / h, 1e-20));
    }

    return {
      hValues,
      forwardErrors,
      centeredErrors,
      richardsonErrors,
      hTheoryO1,
      hTheoryO2,
      hTheoryO4,
    };
  }, [chosen, funcIdx, machineEps]);

  const optimalH_fwd = useMemo(() => {
    const M = chosen.fpp || 1;
    return 2 * Math.sqrt(machineEps / M);
  }, [chosen, machineEps]);

  const optimalH_ctr = useMemo(() => {
    const M = chosen.fpp || 1;
    return Math.pow(3 * machineEps / M, 1 / 3);
  }, [chosen, machineEps]);

  return (
    <SimulationPanel title="Numerical Error vs Step Size h" caption="Truncation error shrinks with h, but roundoff error grows. Watch the two forces fight on a log-log plot. The sweet spot is the V-shaped minimum.">
      <SimulationSettings>
        <div>
          <SimulationLabel>Function</SimulationLabel>
          <SimulationToggle
            options={FUNCTIONS.map((fn, i) => ({ label: fn.label, value: String(i) }))}
            value={String(funcIdx)}
            onChange={(v) => setFuncIdx(Number(v))}
          />
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Machine epsilon: 10^{epsExponent} = {machineEps.toExponential(1)}
          </SimulationLabel>
          <Slider
            value={[epsExponent]}
            onValueChange={([v]) => setEpsExponent(v)}
            min={-16}
            max={-4}
            step={1}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasChart
          data={[
            {
              x: data.hValues,
              y: data.forwardErrors,
              type: 'scatter',
              mode: 'lines+markers',
              line: { color: '#3b82f6', width: 2 },
              marker: { size: 3 },
              name: 'Forward O(h)',
            },
            {
              x: data.hValues,
              y: data.centeredErrors,
              type: 'scatter',
              mode: 'lines+markers',
              line: { color: '#10b981', width: 2 },
              marker: { size: 3 },
              name: 'Centered O(h\u00B2)',
            },
            {
              x: data.hValues,
              y: data.richardsonErrors,
              type: 'scatter',
              mode: 'lines+markers',
              line: { color: '#f59e0b', width: 2 },
              marker: { size: 3 },
              name: 'Richardson O(h\u2074)',
            },
            {
              x: data.hValues,
              y: data.hTheoryO1,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#3b82f6', width: 1, dash: 'dash' },
              name: 'Theory O(h)',
            },
            {
              x: data.hValues,
              y: data.hTheoryO2,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#10b981', width: 1, dash: 'dash' },
              name: 'Theory O(h\u00B2)',
            },
            {
              x: data.hValues,
              y: data.hTheoryO4,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#f59e0b', width: 1, dash: 'dash' },
              name: 'Theory O(h\u2074)',
            },
          ]}
          layout={{
            title: { text: `Error for f'(x) of ${chosen.name}` },
            xaxis: { title: { text: 'Step size h' }, type: 'log' },
            yaxis: { title: { text: 'Absolute error' }, type: 'log' },
            showlegend: true,
            margin: { t: 40, r: 20, b: 50, l: 60 },
          }}
          style={{ width: '100%', height: 420 }}
        />
      </SimulationMain>

      <SimulationResults>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Machine eps</div>
            <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
              {machineEps.toExponential(1)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">h* (forward)</div>
            <div className="text-sm font-mono font-semibold text-[var(--accent)]">
              {optimalH_fwd.toExponential(1)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">h* (centered)</div>
            <div className="text-sm font-mono font-semibold text-[var(--accent)]">
              {optimalH_ctr.toExponential(1)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">True f&apos;(x)</div>
            <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
              {chosen.fprime.toFixed(6)}
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
