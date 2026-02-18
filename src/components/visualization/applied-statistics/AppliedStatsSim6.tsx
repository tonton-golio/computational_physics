'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

// Gaussian PDF
function gaussPdf(x: number, mu: number, sigma: number): number {
  return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mu) / sigma) ** 2);
}

// Log-likelihood for normal distribution
function logLikelihood(sample: number[], mu: number, sigma: number): number {
  let ll = 0;
  for (const x of sample) {
    const pdf = gaussPdf(x, mu, sigma);
    ll += Math.log(Math.max(pdf, 1e-300));
  }
  return ll;
}

// Golden section minimization
function goldenSectionMin(
  f: (x: number) => number, a: number, b: number, tol: number = 1e-5, maxItr: number = 1000
): { x: number; nCalls: number } {
  const phi = (Math.sqrt(5) + 1) / 2;
  let nCalls = 0;
  while (Math.abs(a - b) > tol && nCalls < maxItr) {
    const c = b - (b - a) / phi;
    const d = a + (b - a) / phi;
    if (f(c) > f(d)) {
      a = c;
    } else {
      b = d;
    }
    nCalls++;
  }
  return { x: (a + b) / 2, nCalls };
}

/**
 * Maximum Likelihood Estimation demo.
 * Draws a sample from N(mu, sigma), then uses linear search and golden-section
 * minimization to find the MLE for mu and sigma.
 */
export default function AppliedStatsSim6({ }: SimulationProps) {
  const [muTrue, setMuTrue] = useState(0);
  const [sigTrue, setSigTrue] = useState(1);
  const [sampleSize, setSampleSize] = useState(120);
  const [seed, setSeed] = useState(42);
  const { mergeLayout } = usePlotlyTheme();

  const result = useMemo(() => {
    // Seeded random
    let s = seed;
    const seededRandom = () => {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };
    const seededNormal = () => {
      const u1 = seededRandom();
      const u2 = seededRandom();
      return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    };

    // Generate sample
    const sample: number[] = [];
    for (let i = 0; i < sampleSize; i++) {
      sample.push(muTrue + sigTrue * seededNormal());
    }

    // Linear search for mu
    const nSearch = 500;
    const muRange: number[] = [];
    const muLL: number[] = [];
    for (let i = 0; i < nSearch; i++) {
      const mu = -4 + (8 * i) / (nSearch - 1);
      muRange.push(mu);
      muLL.push(logLikelihood(sample, mu, 1));
    }

    // Find best mu from linear search
    let bestMuLinIdx = 0;
    for (let i = 1; i < nSearch; i++) {
      if (muLL[i] > muLL[bestMuLinIdx]) bestMuLinIdx = i;
    }
    const muBestLinear = muRange[bestMuLinIdx];

    // Linear search for sigma
    const sigRange: number[] = [];
    const sigLL: number[] = [];
    for (let i = 0; i < nSearch; i++) {
      const sig = 0.1 + (5 * i) / (nSearch - 1);
      sigRange.push(sig);
      sigLL.push(logLikelihood(sample, muBestLinear, sig));
    }
    let bestSigLinIdx = 0;
    for (let i = 1; i < nSearch; i++) {
      if (sigLL[i] > sigLL[bestSigLinIdx]) bestSigLinIdx = i;
    }
    const sigBestLinear = sigRange[bestSigLinIdx];

    // Golden section search
    const fMuNeg = (mu: number) => -logLikelihood(sample, mu, 1);
    const { x: muBest, nCalls: nCallsMu } = goldenSectionMin(fMuNeg, -4, 4);
    const fSigNeg = (sig: number) => -logLikelihood(sample, muBest, sig);
    const { x: sigBest, nCalls: nCallsSig } = goldenSectionMin(fSigNeg, 0.1, 5);
    const bestLL = logLikelihood(sample, muBest, sigBest);

    // PDF curve for best fit
    const xMin = Math.min(...sample) - 1;
    const xMax = Math.max(...sample) + 1;
    const xPdf: number[] = [];
    const yPdf: number[] = [];
    for (let i = 0; i <= 200; i++) {
      const xi = xMin + ((xMax - xMin) * i) / 200;
      xPdf.push(xi);
      yPdf.push(gaussPdf(xi, muBest, sigBest));
    }

    return {
      sample,
      xPdf, yPdf,
      muRange, muLL,
      sigRange, sigLL,
      muBestLinear, sigBestLinear,
      muBest, sigBest, bestLL,
      nCallsMu, nCallsSig,
    };
  }, [muTrue, sigTrue, sampleSize, seed]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Maximum Likelihood Estimation</h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        Draw samples from a normal distribution, then find the best-fit parameters via maximum likelihood.
        The left plot shows the sample histogram overlaid with the MLE-fitted Gaussian PDF.
        The right plot shows the log-likelihood as a function of the parameter being optimized.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="text-sm text-[var(--text-muted)]">True mu: {muTrue.toFixed(1)}</label>
          <Slider min={-3} max={3} step={0.1} value={[muTrue]}
            onValueChange={([v]) => setMuTrue(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">True sigma: {sigTrue.toFixed(1)}</label>
          <Slider min={0.3} max={4} step={0.1} value={[sigTrue]}
            onValueChange={([v]) => setSigTrue(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Sample size: {sampleSize}</label>
          <Slider min={10} max={2000} step={10} value={[sampleSize]}
            onValueChange={([v]) => setSampleSize(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Seed: {seed}</label>
          <Slider min={1} max={200} step={1} value={[seed]}
            onValueChange={([v]) => setSeed(v)} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm text-[var(--text-muted)] mb-4">
        <div>
          <strong className="text-[var(--text-strong)]">Golden-section search:</strong> mu = {result.muBest.toFixed(4)}, sigma = {result.sigBest.toFixed(4)}
          <br/>Calls: mu={result.nCallsMu}, sig={result.nCallsSig}
        </div>
        <div>
          <strong className="text-[var(--text-strong)]">Linear search:</strong> mu = {result.muBestLinear.toFixed(4)}, sigma = {result.sigBestLinear.toFixed(4)}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[
            {
              x: result.sample,
              type: 'histogram',
              histnorm: 'probability density',
              nbinsx: Math.max(15, Math.floor(sampleSize ** 0.5)),
              name: 'Sample',
              marker: { color: 'rgba(239,68,68,0.5)' },
            } as any,
            {
              x: result.xPdf,
              y: result.yPdf,
              type: 'scatter',
              mode: 'lines',
              line: { color: 'cyan', width: 2, dash: 'dash' },
              name: 'MLE fit',
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Sample & MLE Fit' },
            margin: { t: 40, r: 20, b: 50, l: 50 },
            xaxis: { title: { text: 'Value' } },
            yaxis: { title: { text: 'Density' } },
            legend: {},
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 400 }}
        />
        <Plot
          data={[
            {
              x: result.muRange,
              y: result.muLL,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#3b82f6', width: 2 },
              name: 'LL(mu)',
            },
            {
              x: result.sigRange,
              y: result.sigLL,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#f59e0b', width: 2 },
              name: 'LL(sigma)',
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Log-Likelihood Curves' },
            margin: { t: 40, r: 20, b: 50, l: 50 },
            xaxis: { title: { text: 'Parameter value' } },
            yaxis: { title: { text: 'Log-likelihood' } },
            legend: {},
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 400 }}
        />
      </div>
    </div>
  );
}
