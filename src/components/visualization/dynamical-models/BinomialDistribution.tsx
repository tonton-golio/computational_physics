'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

/** Log-space factorial using Stirling-like summation for large n */
function logFactorial(n: number): number {
  if (n <= 1) return 0;
  let s = 0;
  for (let i = 2; i <= n; i++) s += Math.log(i);
  return s;
}

/** Binomial PMF: P(k; N, p) = C(N,k) * p^k * (1-p)^(N-k) */
function binomialPMF(k: number, N: number, p: number): number {
  if (k < 0 || k > N) return 0;
  if (p === 0) return k === 0 ? 1 : 0;
  if (p === 1) return k === N ? 1 : 0;
  const logP = logFactorial(N) - logFactorial(k) - logFactorial(N - k)
    + k * Math.log(p) + (N - k) * Math.log(1 - p);
  return Math.exp(logP);
}

export default function BinomialDistribution() {
  const [N1, setN1] = useState(10);
  const [p1, setP1] = useState(0.17);
  const [N2, setN2] = useState(20);
  const [p2, setP2] = useState(0.33);

  const { kVals, y1, y2 } = useMemo(() => {
    const kMax = Math.max(N1, N2);
    const kVals: number[] = [];
    const y1: number[] = [];
    const y2: number[] = [];

    for (let k = 0; k <= kMax; k++) {
      kVals.push(k);
      y1.push(binomialPMF(k, N1, p1));
      y2.push(binomialPMF(k, N2, p2));
    }

    return { kVals, y1, y2 };
  }, [N1, p1, N2, p2]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Binomial Distribution</h3>

      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="text-white text-sm">Distribution 1: N = {N1}</label>
          <input type="range" min={1} max={50} step={1} value={N1}
            onChange={(e) => setN1(parseInt(e.target.value))} className="w-full" />
          <label className="text-white text-sm">p = {p1.toFixed(2)}</label>
          <input type="range" min={0.01} max={1} step={0.01} value={p1}
            onChange={(e) => setP1(parseFloat(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-white text-sm">Distribution 2: N = {N2}</label>
          <input type="range" min={1} max={50} step={1} value={N2}
            onChange={(e) => setN2(parseInt(e.target.value))} className="w-full" />
          <label className="text-white text-sm">p = {p2.toFixed(2)}</label>
          <input type="range" min={0.01} max={1} step={0.01} value={p2}
            onChange={(e) => setP2(parseFloat(e.target.value))} className="w-full" />
        </div>
      </div>

      <Plot
        data={[
          {
            x: kVals, y: y1, type: 'scatter', mode: 'lines+markers',
            marker: { size: 5, color: '#f97316' },
            line: { color: '#f97316', width: 2 },
            name: `N=${N1}, p=${p1.toFixed(2)}`,
          },
          {
            x: kVals, y: y2, type: 'scatter', mode: 'lines+markers',
            marker: { size: 5, color: '#3b82f6' },
            line: { color: '#3b82f6', width: 2 },
            name: `N=${N2}, p=${p2.toFixed(2)}`,
          },
        ] as any}
        layout={{
          height: 400,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          margin: { t: 40, b: 50, l: 60, r: 20 },
          title: { text: 'Binomial Distribution P_N(k)', font: { color: '#9ca3af', size: 14 } },
          xaxis: {
            title: { text: 'k' },
            range: [0, 50],
            gridcolor: 'rgba(75,75,100,0.3)',
            zerolinecolor: 'rgba(75,75,100,0.5)',
          },
          yaxis: {
            title: { text: 'P(k)' },
            range: [0, 1],
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
          The <strong className="text-gray-300">binomial distribution</strong> gives the probability of observing exactly <em>k</em> successes
          in <em>N</em> independent trials, each with success probability <em>p</em>.
          Mean = Np, Variance = Np(1-p).
        </p>
      </div>
    </div>
  );
}
