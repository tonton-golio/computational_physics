'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function gaussianPdf(x: number, mean: number, std: number): number {
  const z = (x - mean) / std;
  return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
}

export default function PriorLikelihoodPosterior({ id: _id }: SimulationComponentProps) {
  const [priorMean, setPriorMean] = useState(0);
  const [priorStd, setPriorStd] = useState(2);
  const [dataValue, setDataValue] = useState(3);
  const [noiseStd, setNoiseStd] = useState(1);

  const result = useMemo(() => {
    // Gaussian conjugate update (analytical)
    const priorPrec = 1 / (priorStd * priorStd);
    const likePrec = 1 / (noiseStd * noiseStd);
    const postPrec = priorPrec + likePrec;
    const postVar = 1 / postPrec;
    const postStd = Math.sqrt(postVar);
    const postMean = (priorPrec * priorMean + likePrec * dataValue) / postPrec;

    // Epsilon ratio connecting to Tikhonov
    const epsilonRatio = noiseStd / priorStd;

    // Generate x range covering all distributions
    const allMeans = [priorMean, dataValue, postMean];
    const allStds = [priorStd, noiseStd, postStd];
    const lo = Math.min(...allMeans.map((m, i) => m - 4 * allStds[i]));
    const hi = Math.max(...allMeans.map((m, i) => m + 4 * allStds[i]));
    const nPts = 300;
    const step = (hi - lo) / (nPts - 1);

    const xArr: number[] = [];
    const priorY: number[] = [];
    const likeY: number[] = [];
    const postY: number[] = [];

    for (let i = 0; i < nPts; i++) {
      const x = lo + i * step;
      xArr.push(x);
      priorY.push(gaussianPdf(x, priorMean, priorStd));
      likeY.push(gaussianPdf(x, dataValue, noiseStd));
      postY.push(gaussianPdf(x, postMean, postStd));
    }

    return { xArr, priorY, likeY, postY, postMean, postStd, epsilonRatio };
  }, [priorMean, priorStd, dataValue, noiseStd]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Prior-Likelihood-Posterior Update</h3>
      <p className="text-[var(--text-muted)] text-sm mb-4">
        1D Gaussian conjugate update. The posterior (green) is the product of the prior (blue) and likelihood (red),
        renormalized. The MAP estimate (yellow line) is the posterior peak.
      </p>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="text-[var(--text-muted)] text-sm">Prior mean: {priorMean.toFixed(1)}</label>
          <Slider
            min={-5} max={5} step={0.1} value={[priorMean]}
            onValueChange={([v]) => setPriorMean(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-sm">Prior std: {priorStd.toFixed(1)}</label>
          <Slider
            min={0.3} max={5} step={0.1} value={[priorStd]}
            onValueChange={([v]) => setPriorStd(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-sm">Data value: {dataValue.toFixed(1)}</label>
          <Slider
            min={-5} max={5} step={0.1} value={[dataValue]}
            onValueChange={([v]) => setDataValue(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-sm">Noise std: {noiseStd.toFixed(1)}</label>
          <Slider
            min={0.3} max={5} step={0.1} value={[noiseStd]}
            onValueChange={([v]) => setNoiseStd(v)}
            className="w-full"
          />
        </div>
      </div>

      <CanvasChart
        data={[
          {
            x: result.xArr,
            y: result.priorY,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#60a5fa', width: 2, dash: 'dash' },
            name: 'Prior',
          },
          {
            x: result.xArr,
            y: result.likeY,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#f87171', width: 2, dash: 'dash' },
            name: 'Likelihood',
          },
          {
            x: result.xArr,
            y: result.postY,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#4ade80', width: 3 },
            name: 'Posterior',
          },
        ]}
        layout={{
          title: { text: 'Bayesian Update' },
          xaxis: { title: { text: 'Model parameter m' } },
          yaxis: { title: { text: 'Density' } },
          height: 400,
          showlegend: true,
          legend: { x: 0.02, y: 0.98 },
          margin: { t: 40, b: 50, l: 60, r: 20 },
          shapes: [
            {
              type: 'line' as const,
              x0: result.postMean,
              x1: result.postMean,
              y0: 0,
              y1: 1,
              yref: 'paper',
              line: { color: '#facc15', width: 2, dash: 'dot' },
            },
          ],
        }}
        style={{ width: '100%' }}
      />

      <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
        <div className="text-[var(--text-muted)]">
          MAP estimate: <span className="text-[var(--text-strong)] font-mono">{result.postMean.toFixed(3)}</span>
        </div>
        <div className="text-[var(--text-muted)]">
          Posterior std: <span className="text-[var(--text-strong)] font-mono">{result.postStd.toFixed(3)}</span>
        </div>
        <div className="text-[var(--text-muted)]">
          Effective ε ratio: <span className="text-[var(--text-strong)] font-mono">{result.epsilonRatio.toFixed(3)}</span>
        </div>
      </div>
      <p className="text-[var(--text-soft)] text-xs mt-2">
        The ε ratio (noise_std / prior_std) corresponds to the Tikhonov regularization parameter.
        Small ε: data dominates. Large ε: prior dominates.
      </p>
    </div>
  );
}
