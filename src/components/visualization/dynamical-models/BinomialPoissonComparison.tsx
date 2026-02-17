'use client';

import { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function factorial(n: number): number {
  if (n <= 1) return 1;
  let out = 1;
  for (let i = 2; i <= n; i++) out *= i;
  return out;
}

function choose(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  return factorial(n) / (factorial(k) * factorial(n - k));
}

export default function BinomialPoissonComparison() {
  const [n, setN] = useState(40);
  const [mean, setMean] = useState(6);

  const { ks, binom, poisson } = useMemo(() => {
    const p = mean / n;
    const maxK = Math.max(20, Math.ceil(mean * 2.8));
    const ks = Array.from({ length: maxK + 1 }, (_, i) => i);
    const binom = ks.map((k) => choose(n, k) * p ** k * (1 - p) ** (n - k));
    const poisson = ks.map((k) => Math.exp(-mean) * mean ** k / factorial(k));
    return { ks, binom, poisson };
  }, [n, mean]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8 space-y-4">
      <h3 className="text-xl font-semibold text-white">Binomial vs Poisson Limit</h3>
      <p className="text-sm text-gray-400">
        Compare exact Binomial(n, p) with Poisson(lambda) where lambda = n p.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-gray-300">n (trials): {n}</label>
          <input className="w-full" type="range" min={10} max={200} step={5} value={n} onChange={(e) => setN(parseInt(e.target.value, 10))} />
        </div>
        <div>
          <label className="text-sm text-gray-300">lambda = n p: {mean.toFixed(1)}</label>
          <input className="w-full" type="range" min={1} max={20} step={0.5} value={mean} onChange={(e) => setMean(parseFloat(e.target.value))} />
        </div>
      </div>
      <Plot
        data={[
          {
            x: ks,
            y: binom,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#3b82f6', width: 2 },
            marker: { size: 5 },
            name: 'Binomial',
          },
          {
            x: ks,
            y: poisson,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#ec4899', width: 2, dash: 'dot' },
            marker: { size: 5 },
            name: 'Poisson approximation',
          },
        ]}
        layout={{
          title: { text: 'Distribution comparison' },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          margin: { t: 40, r: 20, b: 40, l: 50 },
          xaxis: { title: { text: 'k events' }, gridcolor: '#1e1e2e' },
          yaxis: { title: { text: 'Probability' }, gridcolor: '#1e1e2e' },
          height: 430,
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: 430 }}
      />
    </div>
  );
}
