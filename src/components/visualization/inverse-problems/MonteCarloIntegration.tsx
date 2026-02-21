'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';


function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

export default function MonteCarloIntegration({ id }: SimulationComponentProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [nPointsExp, setNPointsExp] = useState(10);
  const [nDim, setNDim] = useState(2);
  const [pNorm, setPNorm] = useState(2.0);
  const nPoints = Math.pow(2, nPointsExp);

  const result = useMemo(() => {
    const rng = seededRandom(42);

    // Generate random points in [-1, 1]^nDim
    const points: number[][] = [];
    for (let i = 0; i < nPoints; i++) {
      const pt: number[] = [];
      for (let d = 0; d < nDim; d++) {
        pt.push(rng() * 2 - 1);
      }
      points.push(pt);
    }

    // Compute p-norm for each point and classify
    const norms: number[] = [];
    const inside: boolean[] = [];
    for (const pt of points) {
      let normVal = 0;
      for (const c of pt) {
        normVal += Math.pow(Math.abs(c), pNorm);
      }
      const finalNorm = normVal; // compare norm^p to 1^p = 1
      norms.push(finalNorm);
      inside.push(finalNorm <= 1);
    }

    const nInside = inside.filter(Boolean).length;
    const fraction = nInside / nPoints;

    // Volume of unit hypersphere: V = 2^nDim * fraction (since cube volume = 2^nDim)
    const cubeVolume = Math.pow(2, nDim);
    const sphereVolume = fraction * cubeVolume;

    // In 2D with p=2, the sphere is a circle with volume pi, so pi estimate:
    const piEstimate = nDim === 2 && Math.abs(pNorm - 2) < 0.01 ? fraction * 4 : null;

    // For plotting, only use first 2 dims
    const xInside: number[] = [];
    const yInside: number[] = [];
    const xOutside: number[] = [];
    const yOutside: number[] = [];

    const maxPlotPoints = Math.min(nPoints, 5000);
    for (let i = 0; i < maxPlotPoints; i++) {
      if (inside[i]) {
        xInside.push(points[i][0]);
        yInside.push(points[i][1]);
      } else {
        xOutside.push(points[i][0]);
        yOutside.push(points[i][1]);
      }
    }

    // Theoretical volume fraction for different dimensions (p=2 norm)
    const dimRange: number[] = [];
    const theoreticalFrac: number[] = [];
    for (let d = 2; d <= 10; d++) {
      dimRange.push(d);
      // Volume of unit ball in d dimensions: pi^(d/2) / Gamma(d/2 + 1)
      // Fraction = V_ball / V_cube = pi^(d/2) / (2^d * Gamma(d/2 + 1))
      const halfD = d / 2;
      let gamma: number;
      if (d % 2 === 0) {
        // Gamma(n+1) = n!
        let fac = 1;
        for (let k = 1; k < halfD; k++) fac *= k;
        gamma = fac;
      } else {
        // Gamma(n+0.5) = (2n)! * sqrt(pi) / (4^n * n!)
        const n = Math.floor(halfD);
        let fac2n = 1;
        for (let k = 1; k <= 2 * n; k++) fac2n *= k;
        let facn = 1;
        for (let k = 1; k <= n; k++) facn *= k;
        gamma = fac2n * Math.sqrt(Math.PI) / (Math.pow(4, n) * facn);
      }
      const vol = Math.pow(Math.PI, halfD) / gamma;
      theoreticalFrac.push(vol / Math.pow(2, d));
    }

    return {
      xInside, yInside, xOutside, yOutside,
      fraction, sphereVolume, piEstimate,
      dimRange, theoreticalFrac,
      nInside, nPoints,
    };
  }, [nPoints, nDim, pNorm]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Monte Carlo Integration: Hypersphere in Hypercube</h3>
      <p className="text-[var(--text-muted)] text-sm mb-4">
        Estimate the volume of a unit hypersphere by random sampling. Points inside the unit
        ball (under the chosen p-norm) are shown in green; points outside are shown in gold.
        In 2D with p=2, this is a classic way to estimate pi.
      </p>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-[var(--text-muted)] text-sm">Points: 2^{nPointsExp} = {nPoints}</label>
          <Slider
            min={4} max={14} step={1} value={[nPointsExp]}
            onValueChange={([v]) => setNPointsExp(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-sm">Dimensions: {nDim}</label>
          <Slider
            min={2} max={10} step={1} value={[nDim]}
            onValueChange={([v]) => setNDim(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-muted)] text-sm">p-norm: {pNorm.toFixed(1)}</label>
          <Slider
            min={0.5} max={8} step={0.1} value={[pNorm]}
            onValueChange={([v]) => setPNorm(v)}
            className="w-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CanvasChart
          data={[
            {
              x: result.xInside,
              y: result.yInside,
              type: 'scatter' as const,
              mode: 'markers' as const,
              marker: { color: 'rgba(34,197,94,0.4)', size: 3 },
              name: 'Inside',
            },
            {
              x: result.xOutside,
              y: result.yOutside,
              type: 'scatter' as const,
              mode: 'markers' as const,
              marker: { color: 'rgba(250,204,21,0.4)', size: 3 },
              name: 'Outside',
            },
          ]}
          layout={{
            title: { text: `First 2 Dimensions (showing up to 5000 pts)` },
            xaxis: { title: { text: 'x_1' }, range: [-1.1, 1.1], scaleanchor: 'y' },
            yaxis: { title: { text: 'x_2' }, range: [-1.1, 1.1] },
            height: 450,
            legend: { x: 0.02, y: 0.98 },
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[
            {
              x: result.dimRange,
              y: result.theoreticalFrac,
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              line: { color: '#f472b6', dash: 'dash' },
              marker: { color: '#f472b6', size: 6 },
              name: 'Theoretical (p=2)',
            },
            {
              x: [nDim],
              y: [result.fraction],
              type: 'scatter' as const,
              mode: 'markers' as const,
              marker: { color: '#22d3ee', size: 12, symbol: 'star' },
              name: `MC estimate (d=${nDim})`,
            },
          ]}
          layout={{
            title: { text: 'Fraction Inside vs Dimension' },
            xaxis: { title: { text: 'Number of dimensions' } },
            yaxis: { title: { text: 'Fraction inside unit sphere' }, type: 'log' },
            height: 450,
            legend: { x: 0.5, y: 0.98 },
            margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
        <div className="bg-[var(--surface-2)] rounded p-3">
          <div className="text-[var(--text-soft)]">Inside / Total</div>
          <div className="text-[var(--text-strong)] font-mono">{result.nInside} / {result.nPoints}</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded p-3">
          <div className="text-[var(--text-soft)]">Fraction inside</div>
          <div className="text-[var(--text-strong)] font-mono">{result.fraction.toFixed(6)}</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded p-3">
          <div className="text-[var(--text-soft)]">Est. sphere volume</div>
          <div className="text-[var(--text-strong)] font-mono">{result.sphereVolume.toFixed(6)}</div>
        </div>
        {result.piEstimate !== null && (
          <div className="bg-[var(--surface-2)] rounded p-3">
            <div className="text-[var(--text-soft)]">Pi estimate</div>
            <div className="text-[var(--text-strong)] font-mono">{result.piEstimate.toFixed(6)}</div>
          </div>
        )}
      </div>
      <p className="text-[var(--text-soft)] text-xs mt-3">
        As the number of dimensions grows, the volume of the hypersphere shrinks dramatically relative to the cube.
        This is the curse of dimensionality: high-dimensional spaces are mostly empty corners.
      </p>
    </div>
  );
}
