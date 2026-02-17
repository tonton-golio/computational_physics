'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

/** Log-space factorial */
function logFactorial(n: number): number {
  if (n <= 1) return 0;
  let s = 0;
  for (let i = 2; i <= n; i++) s += Math.log(i);
  return s;
}

/** Poisson PMF: P(k; lambda) = lambda^k * exp(-lambda) / k! */
function poissonPMF(k: number, lambda: number): number {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  const logP = k * Math.log(lambda) - lambda - logFactorial(k);
  return Math.exp(logP);
}

export default function PoissonDistribution() {
  const [m1, setM1] = useState(1);
  const [m2, setM2] = useState(10);

  const { kVals, y1, y2 } = useMemo(() => {
    const kMax = 50;
    const kVals: number[] = [];
    const y1: number[] = [];
    const y2: number[] = [];

    for (let k = 0; k <= kMax; k++) {
      kVals.push(k);
      y1.push(poissonPMF(k, m1));
      y2.push(poissonPMF(k, m2));
    }

    return { kVals, y1, y2 };
  }, [m1, m2]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Poisson Distribution</h3>

      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="text-white text-sm">Mean m1: {m1}</label>
          <input type="range" min={1} max={32} step={1} value={m1}
            onChange={(e) => setM1(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-white text-sm">Mean m2: {m2}</label>
          <input type="range" min={1} max={32} step={1} value={m2}
            onChange={(e) => setM2(parseInt(e.target.value))} className="w-full" />
        </div>
      </div>

      <Plot
        data={[
          {
            x: kVals, y: y1, type: 'scatter', mode: 'lines+markers',
            marker: { size: 5, color: '#f97316' },
            line: { color: '#f97316', width: 2 },
            name: `m=${m1}`,
          },
          {
            x: kVals, y: y2, type: 'scatter', mode: 'lines+markers',
            marker: { size: 5, color: '#3b82f6' },
            line: { color: '#3b82f6', width: 2 },
            name: `m=${m2}`,
          },
        ] as any}
        layout={{
          height: 400,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          margin: { t: 40, b: 50, l: 60, r: 20 },
          title: { text: 'Poisson Distribution P_m(k)', font: { color: '#9ca3af', size: 14 } },
          xaxis: {
            title: { text: 'k' },
            range: [0, 50],
            gridcolor: 'rgba(75,75,100,0.3)',
            zerolinecolor: 'rgba(75,75,100,0.5)',
          },
          yaxis: {
            title: { text: 'P(k)' },
            range: [0, 0.4],
            gridcolor: 'rgba(75,75,100,0.3)',
            zerolinecolor: 'rgba(75,75,100,0.5)',
          },
          legend: { x: 0.65, y: 0.95, bgcolor: 'rgba(0,0,0,0.3)', font: { color: '#9ca3af' } },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />

      <div className="mt-3 text-sm text-gray-400">
        <p>
          The <strong className="text-gray-300">Poisson distribution</strong> models the number of events occurring in a fixed interval
          when events happen at a constant average rate. It is the limit of the binomial distribution
          when N is large and p is small, with mean m = Np held constant.
          For a Poisson distribution, Mean = Variance = m.
        </p>
      </div>
    </div>
  );
}
