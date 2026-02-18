'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

const G_CONST = 6.674e-11;
const X_OBS = [2, 4, 6, 8, 10, 12, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 38, 40];
const D_OBS = [
  1.378306e-7, 1.181407e-7, 9.6958217e-8, 7.8558607e-8, 6.3732607e-8, 5.2085891e-8,
  3.5852089e-8, 3.0216871e-8, 2.572369e-8, 2.210488e-8, 1.9160639e-8, 1.674152e-8,
  1.473511e-8, 1.305617e-8, 1.163955e-8, 9.4032977e-9, 8.5137701e-9, 7.7420141e-9,
];

function seededRandom(seed: number): () => number {
  let s = Math.max(1, Math.floor(seed));
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
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

function cumulativeZ(heights: number[]): number[] {
  const zs: number[] = [];
  let sum = 0;
  for (const h of heights) {
    sum += h;
    zs.push(sum);
  }
  return zs;
}

function forwardModel(heights: number[], rhos: number[], xs: number[]): number[] {
  const zBase = cumulativeZ(heights);
  const zTop = [0, ...zBase.slice(0, -1)];
  return xs.map((xj) => {
    let sum = 0;
    const x2 = xj * xj;
    for (let i = 0; i < heights.length; i++) {
      const term = Math.log((zBase[i] * zBase[i] + x2) / (zTop[i] * zTop[i] + x2 + 1e-18));
      sum += rhos[i] * term;
    }
    return G_CONST * sum;
  });
}

function loss(obs: number[], pred: number[], sigma: number): number {
  let s = 0;
  for (let i = 0; i < obs.length; i++) {
    const r = obs[i] - pred[i];
    s += r * r;
  }
  return s / (2 * sigma * sigma);
}

export default function VerticalFaultMCMC({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [nWalkers, setNWalkers] = useState(8);
  const [nSteps, setNSteps] = useState(400);
  const [burnIn, setBurnIn] = useState(120);
  const [proposalH, setProposalH] = useState(220);
  const [proposalRho, setProposalRho] = useState(45);
  const [beta, setBeta] = useState(1.0);
  const [seed, setSeed] = useState(42);
  const { mergeLayout } = usePlotlyTheme();

  const result = useMemo(() => {
    const rng = seededRandom(seed);
    const nSlabs = 5;
    const sigma = 4e-9;

    const lossTraces: number[][] = [];
    const acceptanceRates: number[] = [];
    const walkerBestLoss: number[] = [];
    const posteriorH1: number[] = [];
    const posteriorRho1: number[] = [];

    let globalBestLoss = Number.POSITIVE_INFINITY;
    let globalBestPred = [...D_OBS];
    let globalBestHeights = new Array(nSlabs).fill(0);
    let globalBestRhos = new Array(nSlabs).fill(0);

    for (let w = 0; w < nWalkers; w++) {
      let heights = Array.from({ length: nSlabs }, () => 2000 + rng() * 8000);
      let rhos = Array.from({ length: nSlabs }, () => -2000 + rng() * 4000);

      let pred = forwardModel(heights, rhos, X_OBS);
      let currLoss = loss(D_OBS, pred, sigma);
      const trace: number[] = [currLoss];
      let accepted = 0;

      for (let step = 0; step < nSteps; step++) {
        const proposalHeights = heights.map((h) => clip(h + randn(rng) * proposalH, 2000, 10000));
        const proposalRhos = rhos.map((rho) => clip(rho + randn(rng) * proposalRho, -2000, 2000));
        const predNew = forwardModel(proposalHeights, proposalRhos, X_OBS);
        const newLoss = loss(D_OBS, predNew, sigma);

        let accept = false;
        if (newLoss < currLoss) {
          accept = true;
        } else {
          const p = Math.exp(-(newLoss - currLoss) * beta);
          accept = rng() < p;
        }

        if (accept) {
          heights = proposalHeights;
          rhos = proposalRhos;
          pred = predNew;
          currLoss = newLoss;
          accepted += 1;
        }

        if (step >= burnIn) {
          posteriorH1.push(heights[0]);
          posteriorRho1.push(rhos[0]);
        }

        if (currLoss < globalBestLoss) {
          globalBestLoss = currLoss;
          globalBestPred = [...pred];
          globalBestHeights = [...heights];
          globalBestRhos = [...rhos];
        }

        trace.push(currLoss);
      }

      lossTraces.push(trace);
      acceptanceRates.push(accepted / nSteps);
      walkerBestLoss.push(Math.min(...trace));
    }

    return {
      lossTraces,
      acceptanceRates,
      walkerBestLoss,
      posteriorH1,
      posteriorRho1,
      globalBestPred,
      globalBestHeights,
      globalBestRhos,
      globalBestLoss,
    };
  }, [nWalkers, nSteps, burnIn, proposalH, proposalRho, beta, seed]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Vertical Fault (MCMC Inversion)</h3>
      <p className="text-[var(--text-muted)] text-sm mb-4">
        Infer slab heights and density contrasts from observed gravity-gradient data using Metropolis-Hastings
        sampling. This modernized version preserves the old assignment intent while exposing core MCMC controls.
      </p>

      <div className="grid grid-cols-2 lg:grid-cols-7 gap-3 mb-4">
        <div>
          <label className="text-[var(--text-muted)] text-xs">Walkers: {nWalkers}</label>
          <Slider min={2} max={24} step={1} value={[nWalkers]} onValueChange={([v]) => setNWalkers(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Steps: {nSteps}</label>
          <Slider min={120} max={1200} step={20} value={[nSteps]} onValueChange={([v]) => setNSteps(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Burn-in: {burnIn}</label>
          <Slider min={20} max={500} step={10} value={[burnIn]} onValueChange={([v]) => setBurnIn(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Height step: {proposalH}</label>
          <Slider min={30} max={1200} step={10} value={[proposalH]} onValueChange={([v]) => setProposalH(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Density step: {proposalRho}</label>
          <Slider min={5} max={300} step={5} value={[proposalRho]} onValueChange={([v]) => setProposalRho(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Beta: {beta.toFixed(2)}</label>
          <Slider min={0.2} max={2.0} step={0.05} value={[beta]} onValueChange={([v]) => setBeta(v)} />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-xs">Seed: {seed}</label>
          <Slider min={1} max={300} step={1} value={[seed]} onValueChange={([v]) => setSeed(v)} />
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
            opacity: 0.5,
            line: { width: 1 },
          }))}
          layout={mergeLayout({
            title: { text: 'Loss Traces (all walkers)' },
            xaxis: { title: { text: 'iteration' } },
            yaxis: { title: { text: 'loss' }, type: 'log' },
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
          })}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
        <Plot
          data={[
            {
              x: Array.from({ length: result.acceptanceRates.length }, (_, i) => `w${i + 1}`),
              y: result.acceptanceRates,
              type: 'bar' as const,
              marker: { color: 'rgba(96,165,250,0.8)' },
              name: 'acceptance',
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Acceptance Rate per Walker' },
            yaxis: { title: { text: 'acceptance fraction' }, range: [0, 1] },
            height: 320,
            margin: { t: 40, b: 50, l: 60, r: 20 },
          })}
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
              name: 'observed',
              marker: { color: '#fbbf24', size: 7, symbol: 'x' },
            },
            {
              x: X_OBS,
              y: result.globalBestPred,
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              name: 'best prediction',
              marker: { size: 4 },
              line: { color: '#22d3ee', width: 2 },
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Observed vs Predicted Gravity Gradient' },
            xaxis: { title: { text: 'x-position' } },
            yaxis: { title: { text: 'd(x)' } },
            height: 320,
            margin: { t: 40, b: 50, l: 60, r: 20 },
          })}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />

        <Plot
          data={[
            {
              x: result.posteriorH1,
              y: result.posteriorRho1,
              type: 'histogram2d' as const,
              colorscale: 'Viridis',
              nbinsx: 30,
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Posterior of Slab 1 (height vs density)' },
            xaxis: { title: { text: 'height_1 (m)' } },
            yaxis: { title: { text: 'rho_1 (kg/m^3)' } },
            height: 320,
            margin: { t: 40, b: 50, l: 60, r: 20 },
          })}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 text-sm">
        <div className="bg-[var(--surface-1)] rounded p-3 border border-[var(--border-strong)]">
          <div className="text-[var(--text-soft)]">Best loss</div>
          <div className="text-[var(--text-strong)] font-mono">{result.globalBestLoss.toExponential(3)}</div>
        </div>
        <div className="bg-[var(--surface-1)] rounded p-3 border border-[var(--border-strong)]">
          <div className="text-[var(--text-soft)]">Best slab heights (m)</div>
          <div className="text-[var(--text-strong)] font-mono">{result.globalBestHeights.map((v) => v.toFixed(0)).join(', ')}</div>
        </div>
        <div className="bg-[var(--surface-1)] rounded p-3 border border-[var(--border-strong)]">
          <div className="text-[var(--text-soft)]">Best slab densities (kg/m^3)</div>
          <div className="text-[var(--text-strong)] font-mono">{result.globalBestRhos.map((v) => v.toFixed(0)).join(', ')}</div>
        </div>
      </div>
    </div>
  );
}
