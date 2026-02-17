'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

// Seeded random number generator for reproducibility within a render
function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function ConcentrationBounds({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [nExperiments, setNExperiments] = useState(500);
  const [nSamples, setNSamples] = useState(50);
  const [threshold, setThreshold] = useState(0.15);

  const plotData = useMemo(() => {
    const rng = mulberry32(42);

    // We draw n_samples uniform [0,1] random variables, n_experiments times
    // and compute the sample mean for each experiment.
    // Then we compute P(|mean - 0.5| >= t) empirically and compare to bounds.

    const thresholds: number[] = [];
    for (let t = 0.01; t <= 0.5; t += 0.01) {
      thresholds.push(parseFloat(t.toFixed(3)));
    }

    // Generate experiments
    const means: number[] = [];
    for (let i = 0; i < nExperiments; i++) {
      let sum = 0;
      for (let j = 0; j < nSamples; j++) {
        sum += rng();
      }
      means.push(sum / nSamples);
    }

    const mu = 0.5; // true mean for Uniform[0,1]
    const variance = 1.0 / 12.0; // variance of Uniform[0,1]

    const empirical: number[] = [];
    const hoeffding: number[] = [];
    const markov: number[] = [];
    const chebyshev: number[] = [];

    for (const t of thresholds) {
      // Empirical: P(|mean - mu| >= t)
      const count = means.filter((m) => Math.abs(m - mu) >= t).length;
      empirical.push(count / nExperiments);

      // Hoeffding bound: P(|X_bar - mu| >= t) <= 2 * exp(-2*n*t^2)
      // For bounded [0,1] r.v.s: (b-a)=1
      hoeffding.push(Math.min(1, 2 * Math.exp(-2 * nSamples * t * t)));

      // Markov bound: P(|X - mu| >= t) <= E[|X - mu|] / t
      // E[|X_bar - mu|] <= sqrt(Var(X_bar)) = sqrt(variance/n)
      // But Markov applies to non-negative r.v.: P(Y >= a) <= E[Y]/a
      // Here Y = |X_bar - mu|, E[Y] approx sqrt(2*Var(X_bar)/pi) for normal-ish
      // Simpler: use Markov on (X_bar - mu)^2: P((X_bar-mu)^2 >= t^2) <= E[(X_bar-mu)^2]/t^2 = Var/t^2
      // That's Chebyshev. For pure Markov on the mean itself:
      // P(X_bar >= a) <= E[X_bar]/a = mu/a, only for a > 0
      // We'll use: P(|X_bar - mu| >= t) <= E[X_bar^2] / (mu + t)^2...
      // Actually standard Markov: P(X >= a) <= E[X]/a for X >= 0, a > 0
      // Apply to X_bar (which is non-negative): P(X_bar >= mu + t) <= mu / (mu + t)
      // This only gives one-sided. For the illustration, we use:
      // P(X_bar >= mu + t) <= mu / (mu + t), but this is quite loose
      const markovBound = mu / (mu + t);
      markov.push(Math.min(1, markovBound));

      // Chebyshev bound: P(|X_bar - mu| >= t) <= Var(X_bar) / t^2 = (variance/n) / t^2
      chebyshev.push(Math.min(1, (variance / nSamples) / (t * t)));
    }

    return { thresholds, empirical, hoeffding, markov, chebyshev };
  }, [nExperiments, nSamples]);

  // Vertical line for current threshold slider
  const currentIdx = plotData.thresholds.findIndex(
    (t) => Math.abs(t - threshold) < 0.006
  );
  const currentEmpirical = currentIdx >= 0 ? plotData.empirical[currentIdx] : 0;
  const currentHoeffding = currentIdx >= 0 ? plotData.hoeffding[currentIdx] : 0;
  const currentChebyshev = currentIdx >= 0 ? plotData.chebyshev[currentIdx] : 0;
  const currentMarkov = currentIdx >= 0 ? plotData.markov[currentIdx] : 0;

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">
        Concentration Inequalities: Hoeffding, Markov, Chebyshev
      </h3>
      <p className="text-gray-400 text-sm mb-4">
        Draw <strong>n</strong> samples from Uniform[0,1] and compute sample means across many experiments.
        Compare the empirical tail probability P(|X&#772; - 0.5| &ge; t) against theoretical bounds.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-white text-sm">
            Experiments: {nExperiments}
          </label>
          <input
            type="range"
            min={100}
            max={5000}
            step={100}
            value={nExperiments}
            onChange={(e) => setNExperiments(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-sm">
            Samples per experiment (n): {nSamples}
          </label>
          <input
            type="range"
            min={5}
            max={500}
            step={5}
            value={nSamples}
            onChange={(e) => setNSamples(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-sm">
            Threshold (t): {threshold.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.01}
            max={0.5}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <div className="mb-4 text-sm text-gray-300 grid grid-cols-2 md:grid-cols-4 gap-2">
        <div>
          Empirical: <span className="text-blue-400 font-mono">{currentEmpirical.toFixed(4)}</span>
        </div>
        <div>
          Hoeffding: <span className="text-green-400 font-mono">{currentHoeffding.toFixed(4)}</span>
        </div>
        <div>
          Chebyshev: <span className="text-yellow-400 font-mono">{currentChebyshev.toFixed(4)}</span>
        </div>
        <div>
          Markov: <span className="text-red-400 font-mono">{currentMarkov.toFixed(4)}</span>
        </div>
      </div>

      <Plot
        data={[
          {
            x: plotData.thresholds,
            y: plotData.empirical,
            type: 'scatter',
            mode: 'lines',
            name: 'Empirical P(|X&#772;-&mu;|&ge;t)',
            line: { color: '#60a5fa', width: 2 },
          },
          {
            x: plotData.thresholds,
            y: plotData.hoeffding,
            type: 'scatter',
            mode: 'lines',
            name: 'Hoeffding bound',
            line: { color: '#4ade80', width: 2, dash: 'dash' },
          },
          {
            x: plotData.thresholds,
            y: plotData.chebyshev,
            type: 'scatter',
            mode: 'lines',
            name: 'Chebyshev bound',
            line: { color: '#facc15', width: 2, dash: 'dot' },
          },
          {
            x: plotData.thresholds,
            y: plotData.markov,
            type: 'scatter',
            mode: 'lines',
            name: 'Markov bound',
            line: { color: '#f87171', width: 2, dash: 'dashdot' },
          },
          {
            x: [threshold, threshold],
            y: [0, 1],
            type: 'scatter',
            mode: 'lines',
            name: 'Current t',
            line: { color: 'rgba(255,255,255,0.3)', width: 1, dash: 'dot' },
            showlegend: false,
          },
        ]}
        layout={{
          title: { text: 'Concentration Bounds vs Empirical Tail Probability' },
          xaxis: { title: { text: 'Threshold t' }, color: '#9ca3af' },
          yaxis: {
            title: { text: 'Probability' },
            type: 'log',
            range: [-4, 0],
            color: '#9ca3af',
          },
          height: 450,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          legend: { x: 0.6, y: 1 },
          margin: { t: 40, b: 60, l: 60, r: 20 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
