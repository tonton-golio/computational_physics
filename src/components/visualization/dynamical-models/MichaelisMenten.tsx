'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

/**
 * Michaelis-Menten / Monod growth curve:
 *   v = V_max * [S] / (K_m + [S])
 *
 * Used here in the context of bacterial growth physiology (Monod 1949),
 * where the growth rate lambda depends on the concentration of the
 * limiting nutrient S via the Monod equation.
 */
export default function MichaelisMenten() {
  const [lambdaMax, setLambdaMax] = useState(1.25);
  const [Ks, setKs] = useState(0.5);

  const { xVals, yVals } = useMemo(() => {
    const n = 500;
    const xMax = 10.0;
    const xVals: number[] = [];
    const yVals: number[] = [];

    for (let i = 0; i <= n; i++) {
      const x = (i / n) * xMax;
      xVals.push(x);
      yVals.push(lambdaMax * x / (Ks + x));
    }

    return { xVals, yVals };
  }, [lambdaMax, Ks]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Michaelis-Menten / Monod Growth Curve</h3>

      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Maximum growth rate (lambda_max): {lambdaMax.toFixed(2)}</label>
          <Slider value={[lambdaMax]} onValueChange={([v]) => setLambdaMax(v)} min={0.1} max={2.0} step={0.05} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Half-saturation constant (K_S): {Ks.toFixed(2)}</label>
          <Slider value={[Ks]} onValueChange={([v]) => setKs(v)} min={0.1} max={10.0} step={0.1} />
        </div>
      </div>

      <CanvasChart
        data={[
          {
            x: xVals, y: yVals, type: 'scatter', mode: 'lines',
            line: { color: '#3b82f6', width: 2.5 },
            name: 'Growth rate',
          },
          // lambda_max horizontal asymptote
          {
            x: [0, 10], y: [lambdaMax, lambdaMax], type: 'scatter', mode: 'lines',
            line: { color: '#6b7280', dash: 'dash', width: 1 },
            showlegend: false,
          },
          // K_S vertical marker
          {
            x: [Ks, Ks], y: [0, lambdaMax * 0.5], type: 'scatter', mode: 'lines',
            line: { color: '#6b7280', dash: 'dash', width: 1 },
            showlegend: false,
          },
          // lambda_max/2 horizontal marker
          {
            x: [0, Ks], y: [lambdaMax * 0.5, lambdaMax * 0.5], type: 'scatter', mode: 'lines',
            line: { color: '#6b7280', dash: 'dash', width: 1 },
            showlegend: false,
          },
          // Annotation points
          {
            x: [0.15], y: [lambdaMax + 0.05],
            type: 'scatter', mode: 'text',
            text: ['lambda_max'],
            textposition: 'top right',
            textfont: { color: '#9ca3af', size: 12 },
            showlegend: false,
          },
          {
            x: [Ks + 0.15], y: [0.05],
            type: 'scatter', mode: 'text',
            text: ['K_S'],
            textposition: 'top right',
            textfont: { color: '#9ca3af', size: 12 },
            showlegend: false,
          },
        ] as any}
        layout={{
          height: 420,
          margin: { t: 40, b: 60, l: 60, r: 20 },
          title: {
            text: 'lambda = lambda_max * S / (K_S + S)',
          },
          xaxis: {
            title: { text: 'S (concentration of limiting nutrient)' },
            range: [0, 10],
          },
          yaxis: {
            title: { text: 'lambda (growth rate)' },
            range: [0, 2.2],
          },
        }}
        style={{ width: '100%' }}
      />

      <div className="mt-3 text-sm text-[var(--text-muted)]">
        <p>
          The <strong className="text-[var(--text-muted)]">Monod equation</strong> describes bacterial growth rate as a function
          of nutrient concentration, analogous to the Michaelis-Menten equation for enzyme kinetics.
          At low substrate concentrations the growth rate increases approximately linearly;
          at high concentrations it saturates at the maximum rate.
          The half-saturation constant K_S is the substrate concentration at which the growth rate
          is half its maximum value.
        </p>
      </div>
    </div>
  );
}
