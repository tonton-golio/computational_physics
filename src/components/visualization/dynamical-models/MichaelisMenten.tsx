'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

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
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Michaelis-Menten / Monod Growth Curve</h3>

      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="text-white text-sm">Maximum growth rate (lambda_max): {lambdaMax.toFixed(2)}</label>
          <input type="range" min={0.1} max={2.0} step={0.05} value={lambdaMax}
            onChange={(e) => setLambdaMax(parseFloat(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-white text-sm">Half-saturation constant (K_S): {Ks.toFixed(2)}</label>
          <input type="range" min={0.1} max={10.0} step={0.1} value={Ks}
            onChange={(e) => setKs(parseFloat(e.target.value))} className="w-full" />
        </div>
      </div>

      <Plot
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
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          margin: { t: 40, b: 60, l: 60, r: 20 },
          title: {
            text: 'lambda = lambda_max * S / (K_S + S)',
            font: { color: '#9ca3af', size: 14 },
          },
          xaxis: {
            title: { text: 'S (concentration of limiting nutrient)' },
            range: [0, 10],
            gridcolor: 'rgba(75,75,100,0.3)',
            zerolinecolor: 'rgba(75,75,100,0.5)',
          },
          yaxis: {
            title: { text: 'lambda (growth rate)' },
            range: [0, 2.2],
            gridcolor: 'rgba(75,75,100,0.3)',
            zerolinecolor: 'rgba(75,75,100,0.5)',
          },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />

      <div className="mt-3 text-sm text-gray-400">
        <p>
          The <strong className="text-gray-300">Monod equation</strong> describes bacterial growth rate as a function
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
