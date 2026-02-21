'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

/**
 * Bathtub dynamics: approach to steady state.
 *
 * Analytical solution for dn/dt = k - Gamma * n  with n(0) = 0:
 *   n(t) = (k / Gamma) * (1 - exp(-Gamma * t))
 *
 * Steady state:  n_ss = k / Gamma
 * Half-life:     t_half = ln(2) / Gamma
 * Time constant: tau = 1 / Gamma
 */
export default function BathtubDynamics() {
  const [k, setK] = useState(10);
  const [gamma, setGamma] = useState(0.2);
  const [showAnalytical, setShowAnalytical] = useState(false);

  const { tVals, yDefault, yFast, ySlow, tMax, ssDefault, ssFast, ssSlow } = useMemo(() => {
    const n = 500;
    const gammaFast = gamma * 2;
    const gammaSlow = gamma / 2;
    const tMax = 5 / gamma;

    const ssDefault = k / gamma;
    const ssFast = k / gammaFast;
    const ssSlow = k / gammaSlow;

    const tVals: number[] = [];
    const yDefault: number[] = [];
    const yFast: number[] = [];
    const ySlow: number[] = [];

    for (let i = 0; i <= n; i++) {
      const t = (i / n) * tMax;
      tVals.push(t);
      yDefault.push(ssDefault * (1 - Math.exp(-gamma * t)));
      yFast.push(ssFast * (1 - Math.exp(-gammaFast * t)));
      ySlow.push(ssSlow * (1 - Math.exp(-gammaSlow * t)));
    }

    return { tVals, yDefault, yFast, ySlow, tMax, ssDefault, ssFast, ssSlow };
  }, [k, gamma]);

  const halfLife = Math.LN2 / gamma;
  const tau = 1 / gamma;
  const yMax = Math.max(ssDefault, ssFast, ssSlow) * 1.15;

  const traces: any[] = [
    // Default curve
    {
      x: tVals, y: yDefault, type: 'scatter', mode: 'lines',
      line: { color: '#3b82f6', width: 2.5 },
      name: `Default (Gamma = ${gamma.toFixed(2)})`,
    },
    // Fast degradation curve
    {
      x: tVals, y: yFast, type: 'scatter', mode: 'lines',
      line: { color: '#22c55e', width: 2.5 },
      name: `Fast (Gamma = ${(gamma * 2).toFixed(2)})`,
    },
    // Slow degradation curve
    {
      x: tVals, y: ySlow, type: 'scatter', mode: 'lines',
      line: { color: '#f97316', width: 2.5 },
      name: `Slow (Gamma = ${(gamma / 2).toFixed(2)})`,
    },
    // Dashed steady-state line for default
    {
      x: [0, tMax], y: [ssDefault, ssDefault], type: 'scatter', mode: 'lines',
      line: { color: '#3b82f6', dash: 'dash', width: 1 },
      showlegend: false,
    },
    // Dashed steady-state line for fast
    {
      x: [0, tMax], y: [ssFast, ssFast], type: 'scatter', mode: 'lines',
      line: { color: '#22c55e', dash: 'dash', width: 1 },
      showlegend: false,
    },
    // Dashed steady-state line for slow
    {
      x: [0, tMax], y: [ssSlow, ssSlow], type: 'scatter', mode: 'lines',
      line: { color: '#f97316', dash: 'dash', width: 1 },
      showlegend: false,
    },
  ];

  if (showAnalytical) {
    // Annotation labels at the steady-state levels
    traces.push(
      {
        x: [tMax * 0.82], y: [ssDefault + yMax * 0.03],
        type: 'scatter', mode: 'text',
        text: [`n_ss = ${ssDefault.toFixed(1)}`],
        textposition: 'top left',
        textfont: { color: '#3b82f6', size: 11 },
        showlegend: false,
      },
      {
        x: [tMax * 0.82], y: [ssFast + yMax * 0.03],
        type: 'scatter', mode: 'text',
        text: [`n_ss = ${ssFast.toFixed(1)}`],
        textposition: 'top left',
        textfont: { color: '#22c55e', size: 11 },
        showlegend: false,
      },
      {
        x: [tMax * 0.82], y: [ssSlow + yMax * 0.03],
        type: 'scatter', mode: 'text',
        text: [`n_ss = ${ssSlow.toFixed(1)}`],
        textposition: 'top left',
        textfont: { color: '#f97316', size: 11 },
        showlegend: false,
      },
    );
  }

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Bathtub Dynamics: Approach to Steady State</h3>

      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Production rate k: {k.toFixed(1)}</label>
          <Slider value={[k]} onValueChange={([v]) => setK(v)} min={0.5} max={20} step={0.5} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Degradation rate Gamma: {gamma.toFixed(2)}</label>
          <Slider value={[gamma]} onValueChange={([v]) => setGamma(v)} min={0.05} max={2.0} step={0.05} />
        </div>
      </div>

      <div className="mb-4">
        <label className="inline-flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showAnalytical}
            onChange={(e) => setShowAnalytical(e.target.checked)}
            className="w-4 h-4 rounded border-[var(--border-strong)] accent-[var(--accent)]"
          />
          Show analytical steady-state labels
        </label>
      </div>

      <CanvasChart
        data={traces as any}
        layout={{
          height: 420,
          margin: { t: 40, b: 60, l: 60, r: 20 },
          title: {
            text: 'n(t) = (k / Gamma) * (1 - exp(-Gamma * t))',
          },
          xaxis: {
            title: { text: 'Time t' },
            range: [0, tMax],
          },
          yaxis: {
            title: { text: 'n(t)' },
            range: [0, yMax],
          },
        }}
        style={{ width: '100%' }}
      />

      <div className="mt-4 grid grid-cols-3 gap-4 text-sm text-[var(--text-muted)]">
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Steady state n_ss</div>
          <div className="text-lg font-mono">{ssDefault.toFixed(2)}</div>
          <div className="text-xs">k / Gamma = {k.toFixed(1)} / {gamma.toFixed(2)}</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Half-life</div>
          <div className="text-lg font-mono">{halfLife.toFixed(2)}</div>
          <div className="text-xs">ln(2) / Gamma</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Time constant tau</div>
          <div className="text-lg font-mono">{tau.toFixed(2)}</div>
          <div className="text-xs">1 / Gamma</div>
        </div>
      </div>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          All three curves approach their respective steady states k/Gamma.
          The <span style={{ color: '#22c55e' }}>fast degradation</span> curve (Gamma = {(gamma * 2).toFixed(2)}) reaches
          a lower steady state of {ssFast.toFixed(1)} but gets there faster,
          while the <span style={{ color: '#f97316' }}>slow degradation</span> curve (Gamma = {(gamma / 2).toFixed(2)}) climbs
          to a higher steady state of {ssSlow.toFixed(1)} but takes longer.
          The <span style={{ color: '#3b82f6' }}>default</span> curve sits in between at {ssDefault.toFixed(1)}.
          Faster degradation means a lower steady state but faster approach.
        </p>
      </div>
    </div>
  );
}
