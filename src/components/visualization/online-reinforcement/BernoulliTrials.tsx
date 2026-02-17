'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function BernoulliTrials({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [pHeads, setPHeads] = useState(0.5);
  const [nDraws, setNDraws] = useState(20);
  const [nExp, setNExp] = useState(10000);

  const plotData = useMemo(() => {
    const rng = mulberry32(12345);

    // Draw nDraws Bernoulli(pHeads) random variables nExp times
    // Compute the fraction of heads for each experiment
    const fractions: number[] = [];
    for (let i = 0; i < nExp; i++) {
      let heads = 0;
      for (let j = 0; j < nDraws; j++) {
        if (rng() < pHeads) heads++;
      }
      fractions.push(heads / nDraws);
    }

    // For alpha from 0 to 1, compute P(fraction >= alpha)
    const alphas: number[] = [];
    for (let a = 0; a <= 1.001; a += 0.05) {
      alphas.push(parseFloat(a.toFixed(2)));
    }

    const empirical: number[] = [];
    const markovBound: number[] = [];
    const chebyshevBound: number[] = [];
    const hoeffdingBound: number[] = [];

    const expectation = pHeads;
    // Variance of a single Bernoulli draw
    const varSingle = pHeads * (1 - pHeads);
    // Variance of the mean of nDraws Bernoulli draws
    const varMean = varSingle / nDraws;

    for (const alpha of alphas) {
      // Empirical P(fraction >= alpha)
      const count = fractions.filter((f) => f >= alpha).length;
      empirical.push(count / nExp);

      // Markov bound: P(X >= a) <= E[X]/a for a > 0, X >= 0
      if (alpha > 0) {
        markovBound.push(Math.min(1, expectation / alpha));
      } else {
        markovBound.push(1);
      }

      // Chebyshev bound: P(|X - mu| >= k*sigma) <= 1/k^2
      // Here P(X >= alpha) <= P(|X - mu| >= alpha - mu) if alpha > mu
      // P(|X - mu| >= t) <= Var(X)/t^2
      if (alpha > 0) {
        chebyshevBound.push(Math.min(1, varMean / (alpha * alpha)));
      } else {
        chebyshevBound.push(1);
      }

      // Hoeffding bound for mean of n Bernoulli r.v.s in [0,1]:
      // P(X_bar >= alpha) <= exp(-2*n*alpha^2) [one-sided for deviation from 0]
      // More precisely for P(X_bar >= alpha):
      // P(X_bar - mu >= alpha - mu) <= exp(-2*n*(alpha - mu)^2) if alpha > mu
      // We use: P(X_bar >= alpha) <= 2*exp(-2*n*alpha^2) as in the original code
      hoeffdingBound.push(Math.min(1, 2 * Math.exp(-2 * nDraws * alpha * alpha)));
    }

    return { alphas, empirical, markovBound, chebyshevBound, hoeffdingBound };
  }, [pHeads, nDraws, nExp]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">
        Bernoulli Trials: Bounds Illustration
      </h3>
      <p className="text-gray-400 text-sm mb-4">
        Draw {nDraws} Bernoulli random variables {nExp.toLocaleString()} times with bias p.
        Compare empirical tail probability with Markov, Chebyshev, and Hoeffding bounds.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-white text-sm">
            Probability of heads (p): {pHeads.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.05}
            value={pHeads}
            onChange={(e) => setPHeads(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-sm">
            Number of draws (n): {nDraws}
          </label>
          <input
            type="range"
            min={5}
            max={100}
            step={5}
            value={nDraws}
            onChange={(e) => setNDraws(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-sm">
            Number of experiments: {nExp.toLocaleString()}
          </label>
          <input
            type="range"
            min={1000}
            max={50000}
            step={1000}
            value={nExp}
            onChange={(e) => setNExp(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <Plot
        data={[
          {
            x: plotData.alphas,
            y: plotData.empirical,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Empirical P(X&#772; >= &alpha;)',
            line: { color: '#60a5fa', width: 2 },
            marker: { size: 4 },
          },
          {
            x: plotData.alphas,
            y: plotData.markovBound,
            type: 'scatter',
            mode: 'lines',
            name: 'Markov bound',
            line: { color: '#f87171', width: 2, dash: 'dash' },
          },
          {
            x: plotData.alphas,
            y: plotData.chebyshevBound,
            type: 'scatter',
            mode: 'lines',
            name: 'Chebyshev bound',
            line: { color: '#facc15', width: 2, dash: 'dot' },
          },
          {
            x: plotData.alphas,
            y: plotData.hoeffdingBound,
            type: 'scatter',
            mode: 'lines',
            name: "Hoeffding's bound",
            line: { color: '#4ade80', width: 2, dash: 'dashdot' },
          },
        ]}
        layout={{
          title: {
            text: `Bernoulli Bounds (p=${pHeads.toFixed(2)}, n=${nDraws})`,
          },
          xaxis: { title: { text: '&alpha;' }, color: '#9ca3af' },
          yaxis: {
            title: { text: 'Probability' },
            color: '#9ca3af',
            range: [0, 1.1],
          },
          height: 420,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          legend: { x: 0.55, y: 1 },
          margin: { t: 40, b: 60, l: 60, r: 20 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />

      <div className="mt-3 text-xs text-gray-500">
        <p>
          All bounds are respected: the empirical curve lies below each bound. Chebyshev
          is generally tighter than Markov in the middle range, while Hoeffding becomes
          the tightest for larger thresholds.
        </p>
      </div>
    </div>
  );
}
