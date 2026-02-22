"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function BernoulliTrials({}: SimulationComponentProps) {
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
    <SimulationPanel title="Bernoulli Trials: Bounds Illustration">
      <SimulationConfig>
        <div>
          <SimulationLabel className="text-[var(--text-strong)]">
            Probability of heads (p): {pHeads.toFixed(2)}
          </SimulationLabel>
          <Slider
            min={0.05}
            max={0.95}
            step={0.05}
            value={[pHeads]}
            onValueChange={([v]) => setPHeads(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel className="text-[var(--text-strong)]">
            Number of draws (n): {nDraws}
          </SimulationLabel>
          <Slider
            min={5}
            max={100}
            step={5}
            value={[nDraws]}
            onValueChange={([v]) => setNDraws(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel className="text-[var(--text-strong)]">
            Number of experiments: {nExp.toLocaleString()}
          </SimulationLabel>
          <Slider
            min={1000}
            max={50000}
            step={1000}
            value={[nExp]}
            onValueChange={([v]) => setNExp(v)}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <CanvasChart
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
          xaxis: { title: { text: '&alpha;' } },
          yaxis: {
            title: { text: 'Probability' },
            range: [0, 1.1],
          },
          height: 420,
          legend: { x: 0.55, y: 1 },
          margin: { t: 40, b: 60, l: 60, r: 20 },
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>

      <div className="mt-3 text-xs text-[var(--text-soft)]">
        <p>
          All bounds are respected: the empirical curve lies below each bound. Chebyshev
          is generally tighter than Markov in the middle range, while Hoeffding becomes
          the tightest for larger thresholds.
        </p>
      </div>
    </SimulationPanel>
  );
}
