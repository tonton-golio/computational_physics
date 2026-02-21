'use client';

import React, { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';

function computeLandau(t: number, a4: number) {
  const a2 = 1;
  const a6 = 1;
  const mRange: number[] = [];
  const F: number[] = [];
  const nPts = 400;

  for (let i = 0; i <= nPts; i++) {
    const m = -2 + (4 * i) / nPts;
    mRange.push(m);
    const m2 = m * m;
    const m4 = m2 * m2;
    const m6 = m4 * m2;
    F.push(a2 * t * m2 + a4 * m4 + a6 * m6);
  }

  // Find minima
  const minima: { m: number; f: number }[] = [];
  for (let i = 1; i < nPts; i++) {
    if (F[i] < F[i - 1] && F[i] < F[i + 1]) {
      minima.push({ m: mRange[i], f: F[i] });
    }
  }

  return { mRange, F, minima };
}

export function LandauFreeEnergy() {
  const [t, setT] = useState(0.0);
  const [a4, setA4] = useState(1.0);

  const data = useMemo(() => computeLandau(t, a4), [t, a4]);

  const traces: Array<{
    x: number[];
    y: number[];
    type: 'scatter';
    mode: 'lines' | 'markers';
    line?: { color: string; width: number };
    marker?: { color: string; size: number };
    name: string;
  }> = [
    {
      x: data.mRange,
      y: data.F,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#8b5cf6', width: 2 },
      name: 'F(m)',
    },
  ];

  if (data.minima.length > 0) {
    traces.push({
      x: data.minima.map((p) => p.m),
      y: data.minima.map((p) => p.f),
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#ef4444', size: 8 },
      name: 'Minima',
    });
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">
            Reduced temperature t: {t.toFixed(2)}
          </label>
          <Slider
            min={-1}
            max={1}
            step={0.02}
            value={[t]}
            onValueChange={([v]) => setT(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">
            a{'\u2084'} coefficient: {a4.toFixed(2)}
          </label>
          <Slider
            min={-1}
            max={1}
            step={0.02}
            value={[a4]}
            onValueChange={([v]) => setA4(v)}
            className="w-full"
          />
        </div>
      </div>

      <CanvasChart
        data={traces}
        layout={{
          title: { text: 'Landau Free Energy F(m)', font: { size: 14 } },
          xaxis: { title: { text: 'Order parameter m' }, range: [-2, 2] },
          yaxis: { title: { text: 'F(m)' }, range: [-2, 4] },
          showlegend: true,
          margin: { t: 40, r: 20, b: 50, l: 60 },
        }}
        style={{ width: '100%', height: 400 }}
      />

      <p className="text-sm text-[var(--text-muted)]">
        {t > 0
          ? 'T > Tc: single minimum at m = 0 (disordered phase).'
          : t < 0 && a4 > 0
            ? 'T < Tc with a\u2084 > 0: double-well (second-order transition, spontaneous symmetry breaking).'
            : t < 0 && a4 < 0
              ? 'T < Tc with a\u2084 < 0: first-order transition — the minima jump discontinuously.'
              : 'At the critical point T = Tc.'}
      </p>
    </div>
  );
}
