"use client";

import { useMemo, useState } from 'react';
import { mulberry32, gaussianPair } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


const G_CONST = 6.674e-11;
const X_OBS = [2, 4, 6, 8, 10, 12, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 38, 40];
const D_OBS = [
  1.378306e-7, 1.181407e-7, 9.6958217e-8, 7.8558607e-8, 6.3732607e-8, 5.2085891e-8,
  3.5852089e-8, 3.0216871e-8, 2.572369e-8, 2.210488e-8, 1.9160639e-8, 1.674152e-8,
  1.473511e-8, 1.305617e-8, 1.163955e-8, 9.4032977e-9, 8.5137701e-9, 7.7420141e-9,
];

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

export default function VerticalFaultMCMC({}: SimulationComponentProps) {
  const [nWalkers, setNWalkers] = useState(8);
  const [nSteps, setNSteps] = useState(400);
  const [burnIn, setBurnIn] = useState(120);
  const [proposalH, setProposalH] = useState(220);
  const [proposalRho, setProposalRho] = useState(45);
  const [beta, setBeta] = useState(1.0);
  const [seed, setSeed] = useState(42);

  const result = useMemo(() => {
    const rng = mulberry32(seed);
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
        const proposalHeights = heights.map((h) => clip(h + gaussianPair(rng)[0] * proposalH, 2000, 10000));
        const proposalRhos = rhos.map((rho) => clip(rho + gaussianPair(rng)[0] * proposalRho, -2000, 2000));
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
    <SimulationPanel title="Vertical Fault (MCMC Inversion)" caption="Infer slab heights and density contrasts from observed gravity-gradient data using Metropolis-Hastings sampling. This modernized version preserves the old assignment intent while exposing core MCMC controls.">
      <SimulationConfig>
        <div className="grid grid-cols-2 lg:grid-cols-7 gap-3">
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Walkers: {nWalkers}</SimulationLabel>
            <Slider min={2} max={24} step={1} value={[nWalkers]} onValueChange={([v]) => setNWalkers(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Steps: {nSteps}</SimulationLabel>
            <Slider min={120} max={1200} step={20} value={[nSteps]} onValueChange={([v]) => setNSteps(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Burn-in: {burnIn}</SimulationLabel>
            <Slider min={20} max={500} step={10} value={[burnIn]} onValueChange={([v]) => setBurnIn(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Height step: {proposalH}</SimulationLabel>
            <Slider min={30} max={1200} step={10} value={[proposalH]} onValueChange={([v]) => setProposalH(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Density step: {proposalRho}</SimulationLabel>
            <Slider min={5} max={300} step={5} value={[proposalRho]} onValueChange={([v]) => setProposalRho(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Beta: {beta.toFixed(2)}</SimulationLabel>
            <Slider min={0.2} max={2.0} step={0.05} value={[beta]} onValueChange={([v]) => setBeta(v)} />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-xs">Seed: {seed}</SimulationLabel>
            <Slider min={1} max={300} step={1} value={[seed]} onValueChange={([v]) => setSeed(v)} />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <CanvasChart
          data={result.lossTraces.map((trace, i) => ({
            x: Array.from({ length: trace.length }, (_, k) => k),
            y: trace,
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: `walker ${i + 1}`,
            opacity: 0.5,
            line: { width: 1 },
          }))}
          layout={{
            title: { text: 'Loss Traces (all walkers)' },
            xaxis: { title: { text: 'iteration' } },
            yaxis: { title: { text: 'loss' }, type: 'log' },
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
            margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          style={{ width: '100%', height: 320 }}
        />
        <CanvasChart
          data={[
            {
              x: Array.from({ length: result.acceptanceRates.length }, (_, i) => `w${i + 1}`),
              y: result.acceptanceRates,
              type: 'bar' as const,
              marker: { color: '#60a5fa' },
              opacity: 0.8,
              name: 'acceptance',
            },
          ]}
          layout={{
            title: { text: 'Acceptance Rate per Walker' },
            yaxis: { title: { text: 'acceptance fraction' }, range: [0, 1] },
            margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          style={{ width: '100%', height: 320 }}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <CanvasChart
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
          layout={{
            title: { text: 'Observed vs Predicted Gravity Gradient' },
            xaxis: { title: { text: 'x-position' } },
            yaxis: { title: { text: 'd(x)' } },
            margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          style={{ width: '100%', height: 320 }}
        />

        {(() => {
          // Manual 2D histogram binning for CanvasHeatmap
          const h1 = result.posteriorH1;
          const rho1 = result.posteriorRho1;
          if (h1.length === 0) return null;
          const nBins = 30;
          const hMin = Math.min(...h1), hMax = Math.max(...h1);
          const rMin = Math.min(...rho1), rMax = Math.max(...rho1);
          const hStep = (hMax - hMin) / nBins || 1;
          const rStep = (rMax - rMin) / nBins || 1;
          const z: number[][] = Array.from({ length: nBins }, () => Array(nBins).fill(0));
          for (let k = 0; k < h1.length; k++) {
            const ix = Math.min(Math.floor((h1[k] - hMin) / hStep), nBins - 1);
            const iy = Math.min(Math.floor((rho1[k] - rMin) / rStep), nBins - 1);
            z[iy][ix]++;
          }
          const xLabels = Array.from({ length: nBins }, (_, i) => hMin + (i + 0.5) * hStep);
          const yLabels = Array.from({ length: nBins }, (_, i) => rMin + (i + 0.5) * rStep);
          return (
            <CanvasHeatmap
              data={[{ z, x: xLabels, y: yLabels, colorscale: 'Viridis' }]}
              layout={{
                title: { text: 'Posterior of Slab 1 (height vs density)' },
                xaxis: { title: { text: 'height_1 (m)' } },
                yaxis: { title: { text: 'rho_1 (kg/m^3)' } },
                margin: { t: 40, b: 50, l: 60, r: 20 },
              }}
              style={{ width: '100%', height: 320 }}
            />
          );
        })()}
      </div>

      </SimulationMain>
      <SimulationResults>
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
      </SimulationResults>
    </SimulationPanel>
  );
}
