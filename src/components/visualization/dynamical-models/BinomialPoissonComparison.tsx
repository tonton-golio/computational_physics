'use client';

import { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

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
  const { mergeLayout } = usePlotlyTheme();

  const { ks, binom, poisson } = useMemo(() => {
    const p = mean / n;
    const maxK = Math.max(20, Math.ceil(mean * 2.8));
    const ks = Array.from({ length: maxK + 1 }, (_, i) => i);
    const binom = ks.map((k) => choose(n, k) * p ** k * (1 - p) ** (n - k));
    const poisson = ks.map((k) => Math.exp(-mean) * mean ** k / factorial(k));
    return { ks, binom, poisson };
  }, [n, mean]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8 space-y-4">
      <h3 className="text-xl font-semibold text-[var(--text-strong)]">Binomial vs Poisson Limit</h3>
      <p className="text-sm text-[var(--text-muted)]">
        Compare exact Binomial(n, p) with Poisson(lambda) where lambda = n p.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">n (trials): {n}</label>
          <Slider value={[n]} onValueChange={([v]) => setN(v)} min={10} max={200} step={5} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">lambda = n p: {mean.toFixed(1)}</label>
          <Slider value={[mean]} onValueChange={([v]) => setMean(v)} min={1} max={20} step={0.5} />
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
        layout={mergeLayout({
          title: { text: 'Distribution comparison' },
          margin: { t: 40, r: 20, b: 40, l: 50 },
          xaxis: { title: { text: 'k events' } },
          yaxis: { title: { text: 'Probability' } },
          height: 430,
        })}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: 430 }}
      />
    </div>
  );
}
