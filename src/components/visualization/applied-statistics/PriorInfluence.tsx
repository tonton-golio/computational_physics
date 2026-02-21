'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

export default function PriorInfluence() {
  const [priorPrecision, setPriorPrecision] = useState(1.0);
  const [priorMean, setPriorMean] = useState(0.0);

  const { xVals, priorY, likelihoodY, posteriorY } = useMemo(() => {
    // Data: 5 observations with mean ~3, precision ~2
    const dataMean = 3.0;
    const dataPrecision = 2.0; // 1/sigma^2 from data
    const nObs = 5;
    const likePrecision = dataPrecision * nObs;
    const likeMean = dataMean;

    // Prior: Normal(priorMean, 1/priorPrecision)
    const postPrecision = priorPrecision + likePrecision;
    const postMean = (priorPrecision * priorMean + likePrecision * likeMean) / postPrecision;

    const xs: number[] = [];
    const prior: number[] = [];
    const like: number[] = [];
    const post: number[] = [];

    for (let i = 0; i <= 200; i++) {
      const x = -4 + i * 0.06;
      xs.push(x);
      // Prior
      const zp = (x - priorMean) * Math.sqrt(priorPrecision);
      prior.push(Math.sqrt(priorPrecision / (2 * Math.PI)) * Math.exp(-0.5 * zp * zp));
      // Likelihood (as function of mu)
      const zl = (x - likeMean) * Math.sqrt(likePrecision);
      like.push(Math.sqrt(likePrecision / (2 * Math.PI)) * Math.exp(-0.5 * zl * zl));
      // Posterior
      const zpost = (x - postMean) * Math.sqrt(postPrecision);
      post.push(Math.sqrt(postPrecision / (2 * Math.PI)) * Math.exp(-0.5 * zpost * zpost));
    }

    return { xVals: xs, priorY: prior, likelihoodY: like, posteriorY: post };
  }, [priorPrecision, priorMean]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Prior Influence on the Posterior</h3>
      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Prior precision: {priorPrecision.toFixed(1)}</label>
          <Slider value={[priorPrecision]} onValueChange={([v]) => setPriorPrecision(v)} min={0.1} max={20} step={0.1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Prior mean: {priorMean.toFixed(1)}</label>
          <Slider value={[priorMean]} onValueChange={([v]) => setPriorMean(v)} min={-4} max={8} step={0.1} />
        </div>
      </div>
      <div className="mb-3 text-sm text-[var(--text-muted)]">
        Data mean: 3.0 (5 observations) | The posterior interpolates between the prior and the likelihood.
      </div>
      <CanvasChart
        data={[
          { x: xVals, y: priorY, type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 2, dash: 'dash' }, name: 'Prior' },
          { x: xVals, y: likelihoodY, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2, dash: 'dot' }, name: 'Likelihood' },
          { x: xVals, y: posteriorY, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 3 }, name: 'Posterior' },
        ]}
        layout={{
          height: 400,
          xaxis: { title: { text: 'mu' } },
          yaxis: { title: { text: 'Density' } },
        }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
