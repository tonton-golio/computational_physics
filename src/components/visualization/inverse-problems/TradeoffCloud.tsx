'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function seededRandom(seed: number): () => number {
  let s = Math.max(1, Math.floor(seed));
  return () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };
}
function randn(rng: () => number): number {
  return Math.sqrt(-2 * Math.log(Math.max(rng(), 1e-12))) * Math.cos(2 * Math.PI * rng());
}
function forward(depth: number, slip: number, xObs: number[]): number[] {
  return xObs.map(x => slip * depth / (depth * depth + x * x));
}
function misfit(dObs: number[], dPred: number[], sigma: number): number {
  let s = 0;
  for (let i = 0; i < dObs.length; i++) { const r = (dObs[i] - dPred[i]) / sigma; s += r * r; }
  return 0.5 * s;
}

export default function TradeoffCloud({ id: _id }: SimulationComponentProps) {
  const [noiseLevel, setNoiseLevel] = useState(0.05);
  const [nSamples, setNSamples] = useState(300);
  const [trueDepth, setTrueDepth] = useState(3.0);
  const trueSlip = 5.0;

  const result = useMemo(() => {
    const rng = seededRandom(42), sigma = Math.max(noiseLevel, 0.005);
    const xObs = Array.from({ length: 15 }, (_, i) => 0.5 + i * 0.7);
    const dObs = forward(trueDepth, trueSlip, xObs).map(d => d + sigma * randn(rng));
    let curD = 2 + rng() * 4, curS = 2 + rng() * 6;
    let curM = misfit(dObs, forward(curD, curS, xObs), sigma);
    const depths: number[] = [], slips: number[] = [], burnIn = 100;
    for (let s = 0; s < burnIn + nSamples; s++) {
      const pD = Math.max(0.5, curD + randn(rng) * 0.3);
      const pS = Math.max(0.1, curS + randn(rng) * 0.5);
      const pM = misfit(dObs, forward(pD, pS, xObs), sigma);
      if (Math.log(rng()) < curM - pM) { curD = pD; curS = pS; curM = pM; }
      if (s >= burnIn) { depths.push(curD); slips.push(curS); }
    }
    const nS = depths.length;
    const mD = depths.reduce((a, b) => a + b, 0) / nS, mS = slips.reduce((a, b) => a + b, 0) / nS;
    let covDS = 0, varD = 0, varS = 0;
    for (let i = 0; i < nS; i++) {
      const dd = depths[i] - mD, ds = slips[i] - mS;
      covDS += dd * ds; varD += dd * dd; varS += ds * ds;
    }
    const corr = covDS / (Math.sqrt(varD * varS) + 1e-12);
    const nBins = 25;
    const dMin = Math.min(...depths), dMax = Math.max(...depths), dSt = (dMax - dMin) / nBins || 0.1;
    const sMin = Math.min(...slips), sMax = Math.max(...slips), sSt = (sMax - sMin) / nBins || 0.1;
    const dBins = Array.from({ length: nBins }, (_, i) => dMin + (i + 0.5) * dSt);
    const sBins = Array.from({ length: nBins }, (_, i) => sMin + (i + 0.5) * sSt);
    const dCounts = new Array(nBins).fill(0), sCounts = new Array(nBins).fill(0);
    for (const d of depths) { const i = Math.min(Math.floor((d - dMin) / dSt), nBins - 1); if (i >= 0) dCounts[i]++; }
    for (const s of slips) { const i = Math.min(Math.floor((s - sMin) / sSt), nBins - 1); if (i >= 0) sCounts[i]++; }
    return { depths, slips, corr, dBins, dCounts, sBins, sCounts };
  }, [noiseLevel, nSamples, trueDepth]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Trade-off Cloud</h3>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Noise: {noiseLevel.toFixed(3)}</label>
          <Slider value={[noiseLevel]} onValueChange={([v]) => setNoiseLevel(v)} min={0.005} max={0.3} step={0.005} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Samples: {nSamples}</label>
          <Slider value={[nSamples]} onValueChange={([v]) => setNSamples(v)} min={100} max={800} step={50} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">True depth: {trueDepth.toFixed(1)}</label>
          <Slider value={[trueDepth]} onValueChange={([v]) => setTrueDepth(v)} min={1.0} max={6.0} step={0.5} />
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <CanvasChart
          data={[
            { x: result.depths, y: result.slips, type: 'scatter', mode: 'markers',
              marker: { size: 3, color: 'rgba(59,130,246,0.4)' }, name: 'MCMC samples' },
            { x: [trueDepth], y: [trueSlip], type: 'scatter', mode: 'markers',
              marker: { size: 10, color: '#ef4444' }, name: 'True values' },
          ]}
          layout={{
            title: { text: `Depth-Slip Trade-off (r = ${result.corr.toFixed(3)})` },
            xaxis: { title: { text: 'Depth' } }, yaxis: { title: { text: 'Slip' } },
            height: 400, margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
        <div className="space-y-2">
          <CanvasChart
            data={[{ x: result.dBins, y: result.dCounts, type: 'bar',
              marker: { color: 'rgba(59,130,246,0.7)' }, name: 'Depth marginal' }]}
            layout={{
              title: { text: 'Depth Marginal' }, xaxis: { title: { text: 'Depth' } },
              yaxis: { title: { text: 'Count' } }, height: 190,
              margin: { t: 30, b: 40, l: 50, r: 15 },
              shapes: [{ type: 'line', x0: trueDepth, x1: trueDepth, y0: 0, y1: 1, yref: 'paper',
                line: { color: '#ef4444', dash: 'dash', width: 1.5 } }],
            }}
            style={{ width: '100%' }}
          />
          <CanvasChart
            data={[{ x: result.sBins, y: result.sCounts, type: 'bar',
              marker: { color: 'rgba(16,185,129,0.7)' }, name: 'Slip marginal' }]}
            layout={{
              title: { text: 'Slip Marginal' }, xaxis: { title: { text: 'Slip' } },
              yaxis: { title: { text: 'Count' } }, height: 190,
              margin: { t: 30, b: 40, l: 50, r: 15 },
              shapes: [{ type: 'line', x0: trueSlip, x1: trueSlip, y0: 0, y1: 1, yref: 'paper',
                line: { color: '#ef4444', dash: 'dash', width: 1.5 } }],
            }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          The scatter cloud is elongated along the depth-slip trade-off direction: deeper faults
          with larger slip produce similar surface data as shallower faults with less slip.
          Increasing noise stretches the cloud. The correlation coefficient r quantifies the
          trade-off strength. Marginal histograms show individual parameter uncertainty.
        </p>
      </div>
    </div>
  );
}
