'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

const X_OBS = [535, 749, 963, 1177, 1391, 1605, 1819, 2033, 2247, 2461, 2675, 2889];
const D_OBS = [-15.1, -23.9, -31.2, -36.9, -40.8, -42.8, -42.5, -40.7, -37.1, -31.5, -21.9, -12.9];

function seededRandom(seed: number): () => number {
  let s = Math.max(1, Math.floor(seed));
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
}

function randn(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clip(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function makeForwardMatrix(xs: number[], kernelWidth: number): number[][] {
  const K: number[][] = [];
  for (let j = 0; j < xs.length; j++) {
    const row: number[] = [];
    for (let i = 0; i < xs.length; i++) {
      const dist = xs[j] - xs[i];
      row.push(Math.exp(-(dist * dist) / (2 * kernelWidth * kernelWidth)));
    }
    const s = row.reduce((a, b) => a + b, 0);
    K.push(row.map((v) => v / (s || 1)));
  }
  return K;
}

function matVecMul(A: number[][], v: number[]): number[] {
  return A.map((row) => row.reduce((sum, aij, i) => sum + aij * v[i], 0));
}

function forward(thickness: number[], K: number[][], scale: number): number[] {
  const smooth = matVecMul(K, thickness);
  return smooth.map((h) => -scale * h);
}

function mseLoss(obs: number[], pred: number[]): number {
  let s = 0;
  for (let i = 0; i < obs.length; i++) {
    const r = obs[i] - pred[i];
    s += r * r;
  }
  return s / obs.length;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = clip(Math.floor((sorted.length - 1) * p), 0, sorted.length - 1);
  return sorted[idx];
}

export default function GlacierThicknessMCMC({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [nWalkers, setNWalkers] = useState(6);
  const [nSteps, setNSteps] = useState(450);
  const [burnIn, setBurnIn] = useState(140);
  const [proposalStd, setProposalStd] = useState(14);
  const [smoothnessWeight, setSmoothnessWeight] = useState(0.15);
  const [kernelWidth, setKernelWidth] = useState(320);
  const [scale, setScale] = useState(0.16);
  const [seed, setSeed] = useState(12);

  const result = useMemo(() => {
    const rng = seededRandom(seed);
    const K = makeForwardMatrix(X_OBS, kernelWidth);

    const lossTraces: number[][] = [];
    const acceptanceRates: number[] = [];
    const posteriorThickness: number[][] = [];
    const posteriorPredictions: number[][] = [];

    let bestLoss = Number.POSITIVE_INFINITY;
    let bestThickness = new Array(X_OBS.length).fill(200);
    let bestPrediction = forward(bestThickness, K, scale);

    const roughness = (h: number[]) => {
      let r = 0;
      for (let i = 1; i < h.length - 1; i++) {
        const curvature = h[i + 1] - 2 * h[i] + h[i - 1];
        r += curvature * curvature;
      }
      return r / h.length;
    };

    for (let w = 0; w < nWalkers; w++) {
      let thickness = Array.from({ length: X_OBS.length }, () => 120 + rng() * 680);
      let pred = forward(thickness, K, scale);
      let currLoss = mseLoss(D_OBS, pred) + smoothnessWeight * roughness(thickness);
      const trace = [currLoss];
      let accepted = 0;

      for (let step = 0; step < nSteps; step++) {
        const proposal = thickness.map((h) => clip(h + randn(rng) * proposalStd, 20, 900));
        const predNew = forward(proposal, K, scale);
        const newLoss = mseLoss(D_OBS, predNew) + smoothnessWeight * roughness(proposal);

        let accept = false;
        if (newLoss < currLoss) {
          accept = true;
        } else {
          const p = Math.exp(-(newLoss - currLoss) * 1.4);
          accept = rng() < p;
        }

        if (accept) {
          thickness = proposal;
          pred = predNew;
          currLoss = newLoss;
          accepted += 1;
        }

        if (step >= burnIn) {
          posteriorThickness.push([...thickness]);
          posteriorPredictions.push([...pred]);
        }

        if (currLoss < bestLoss) {
          bestLoss = currLoss;
          bestThickness = [...thickness];
          bestPrediction = [...pred];
        }

        trace.push(currLoss);
      }

      lossTraces.push(trace);
      acceptanceRates.push(accepted / nSteps);
    }

    const posteriorMeanThickness = bestThickness.map((_, i) => {
      if (posteriorThickness.length === 0) return bestThickness[i];
      return posteriorThickness.reduce((s, arr) => s + arr[i], 0) / posteriorThickness.length;
    });

    const predP10 = X_OBS.map((_, i) => percentile(posteriorPredictions.map((p) => p[i]), 0.1));
    const predP90 = X_OBS.map((_, i) => percentile(posteriorPredictions.map((p) => p[i]), 0.9));

    return {
      lossTraces,
      acceptanceRates,
      posteriorMeanThickness,
      bestThickness,
      bestPrediction,
      bestLoss,
      predP10,
      predP90,
    };
  }, [nWalkers, nSteps, burnIn, proposalStd, smoothnessWeight, kernelWidth, scale, seed]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Glacier Thickness (MCMC Inversion)</h3>
      <p className="text-gray-400 text-sm mb-4">
        Estimate a glacier thickness profile from Bouguer-anomaly data using a simplified forward model and
        Metropolis sampling. The posterior gives both a best-fit profile and uncertainty bounds.
      </p>

      <div className="grid grid-cols-2 lg:grid-cols-8 gap-3 mb-4">
        <div>
          <label className="text-gray-300 text-xs">Walkers: {nWalkers}</label>
          <input type="range" min={2} max={18} step={1} value={nWalkers} onChange={(e) => setNWalkers(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Steps: {nSteps}</label>
          <input type="range" min={160} max={1500} step={20} value={nSteps} onChange={(e) => setNSteps(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Burn-in: {burnIn}</label>
          <input type="range" min={20} max={700} step={10} value={burnIn} onChange={(e) => setBurnIn(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Proposal: {proposalStd}</label>
          <input type="range" min={2} max={80} step={1} value={proposalStd} onChange={(e) => setProposalStd(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Smoothness: {smoothnessWeight.toFixed(2)}</label>
          <input type="range" min={0} max={1} step={0.01} value={smoothnessWeight} onChange={(e) => setSmoothnessWeight(parseFloat(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Kernel width: {kernelWidth}</label>
          <input type="range" min={120} max={700} step={10} value={kernelWidth} onChange={(e) => setKernelWidth(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Scale: {scale.toFixed(2)}</label>
          <input type="range" min={0.04} max={0.4} step={0.01} value={scale} onChange={(e) => setScale(parseFloat(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Seed: {seed}</label>
          <input type="range" min={1} max={200} step={1} value={seed} onChange={(e) => setSeed(parseInt(e.target.value))} className="w-full" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <Plot
          data={result.lossTraces.map((trace, i) => ({
            x: Array.from({ length: trace.length }, (_, k) => k),
            y: trace,
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: `walker ${i + 1}`,
            opacity: 0.55,
            line: { width: 1 },
          }))}
          layout={{
            title: { text: 'Loss Traces (MCMC Walkers)' },
            xaxis: { title: { text: 'iteration' }, color: '#9ca3af' },
            yaxis: { title: { text: 'loss' }, type: 'log', color: '#9ca3af' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            height: 320,
            margin: { t: 40, b: 50, l: 60, r: 20 },
            showlegend: false,
            shapes: [{
              type: 'line',
              x0: burnIn,
              x1: burnIn,
              y0: 0,
              y1: 1,
              yref: 'paper',
              line: { color: '#22d3ee', dash: 'dash' },
            }],
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
        <Plot
          data={[{
            x: Array.from({ length: result.acceptanceRates.length }, (_, i) => `w${i + 1}`),
            y: result.acceptanceRates,
            type: 'bar' as const,
            marker: { color: 'rgba(167,139,250,0.85)' },
          }]}
          layout={{
            title: { text: 'Acceptance Rate per Walker' },
            xaxis: { color: '#9ca3af' },
            yaxis: { title: { text: 'acceptance fraction' }, range: [0, 1], color: '#9ca3af' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            height: 320,
            margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <Plot
          data={[
            {
              x: X_OBS,
              y: D_OBS,
              type: 'scatter' as const,
              mode: 'markers' as const,
              name: 'observed anomaly',
              marker: { color: '#fbbf24', symbol: 'x', size: 8 },
            },
            {
              x: X_OBS,
              y: result.predP90,
              type: 'scatter' as const,
              mode: 'lines' as const,
              line: { color: 'rgba(34,211,238,0.0)' },
              showlegend: false,
              hoverinfo: 'skip' as const,
            },
            {
              x: X_OBS,
              y: result.predP10,
              type: 'scatter' as const,
              mode: 'lines' as const,
              fill: 'tonexty' as const,
              fillcolor: 'rgba(34,211,238,0.18)',
              line: { color: 'rgba(34,211,238,0.0)' },
              name: 'posterior 10-90%',
            },
            {
              x: X_OBS,
              y: result.bestPrediction,
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              name: 'best prediction',
              line: { color: '#22d3ee', width: 2 },
              marker: { size: 4 },
            },
          ]}
          layout={{
            title: { text: 'Observed vs Posterior Prediction' },
            xaxis: { title: { text: 'distance along glacier (m)' }, color: '#9ca3af' },
            yaxis: { title: { text: 'Bouguer anomaly' }, color: '#9ca3af' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            legend: { bgcolor: 'rgba(0,0,0,0.3)' },
            height: 340,
            margin: { t: 40, b: 55, l: 60, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
        <Plot
          data={[
            {
              x: X_OBS,
              y: result.posteriorMeanThickness,
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              name: 'posterior mean thickness',
              line: { color: '#fb7185', width: 2 },
              marker: { size: 5 },
            },
            {
              x: X_OBS,
              y: result.bestThickness,
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: 'best thickness sample',
              line: { color: '#a78bfa', width: 1.5, dash: 'dot' },
            },
          ]}
          layout={{
            title: { text: 'Inferred Glacier Thickness Profile' },
            xaxis: { title: { text: 'distance along glacier (m)' }, color: '#9ca3af' },
            yaxis: { title: { text: 'thickness (m)' }, color: '#9ca3af' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            legend: { bgcolor: 'rgba(0,0,0,0.3)' },
            height: 340,
            margin: { t: 40, b: 55, l: 60, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="bg-[#0a0a15] rounded p-3 text-sm">
        <div className="text-gray-500">Best posterior objective</div>
        <div className="text-white font-mono">{result.bestLoss.toFixed(4)}</div>
      </div>
    </div>
  );
}
