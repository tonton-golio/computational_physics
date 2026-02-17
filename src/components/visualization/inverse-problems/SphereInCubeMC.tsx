'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

function seededRandom(seed: number): () => number {
  let s = Math.max(1, Math.floor(seed));
  return () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647;
  };
}

function gammaHalfInteger(n: number): number {
  if (n <= 0) return 1;
  if (n % 2 === 0) {
    let fac = 1;
    for (let i = 1; i < n / 2; i++) fac *= i;
    return fac;
  }
  const k = Math.floor((n - 1) / 2);
  let fac2k = 1;
  for (let i = 1; i <= 2 * k; i++) fac2k *= i;
  let fack = 1;
  for (let i = 1; i <= k; i++) fack *= i;
  return (fac2k * Math.sqrt(Math.PI)) / (Math.pow(4, k) * fack);
}

function theoreticalVolumeFraction(dim: number): number {
  const volBall = Math.pow(Math.PI, dim / 2) / gammaHalfInteger(dim + 2);
  const volCube = Math.pow(2, dim);
  return volBall / volCube;
}

export default function SphereInCubeMC({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [nDims, setNDims] = useState(4);
  const [nPointsExp, setNPointsExp] = useState(12);
  const [seed, setSeed] = useState(17);
  const nPoints = Math.pow(2, nPointsExp);

  const result = useMemo(() => {
    const rng = seededRandom(seed);
    const insideFlags: boolean[] = [];
    const points2D: Array<[number, number]> = [];

    let insideCount = 0;
    const nPlot = Math.min(nPoints, 6000);

    for (let i = 0; i < nPoints; i++) {
      let norm2 = 0;
      const firstX = rng() * 2 - 1;
      const firstY = rng() * 2 - 1;
      norm2 += firstX * firstX + firstY * firstY;
      for (let d = 2; d < nDims; d++) {
        const x = rng() * 2 - 1;
        norm2 += x * x;
      }
      const inside = norm2 <= 1;
      if (inside) insideCount += 1;

      if (i < nPlot) {
        points2D.push([firstX, firstY]);
        insideFlags.push(inside);
      }
    }

    const frac = insideCount / nPoints;
    const estVolume = frac * Math.pow(2, nDims);

    const xInside: number[] = [];
    const yInside: number[] = [];
    const xOutside: number[] = [];
    const yOutside: number[] = [];
    for (let i = 0; i < points2D.length; i++) {
      if (insideFlags[i]) {
        xInside.push(points2D[i][0]);
        yInside.push(points2D[i][1]);
      } else {
        xOutside.push(points2D[i][0]);
        yOutside.push(points2D[i][1]);
      }
    }

    const dimAxis: number[] = [];
    const theoryAxis: number[] = [];
    for (let d = 2; d <= 12; d++) {
      dimAxis.push(d);
      theoryAxis.push(theoreticalVolumeFraction(d));
    }

    const piEstimate = nDims === 2 ? frac * 4 : null;

    return {
      frac,
      estVolume,
      insideCount,
      xInside,
      yInside,
      xOutside,
      yOutside,
      dimAxis,
      theoryAxis,
      piEstimate,
      nPlot,
    };
  }, [nDims, nPoints, seed]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Sphere in Cube (Monte Carlo Accept/Reject)</h3>
      <p className="text-gray-400 text-sm mb-4">
        Reproduce the classic accept/reject demo: sample points uniformly in [-1,1]^N and keep only points
        inside the unit sphere. This visualizes why high-dimensional volume estimation is difficult.
      </p>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-gray-300 text-xs">Dimensions: {nDims}</label>
          <input type="range" min={2} max={12} step={1} value={nDims} onChange={(e) => setNDims(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Points: 2^{nPointsExp} = {nPoints}</label>
          <input type="range" min={6} max={16} step={1} value={nPointsExp} onChange={(e) => setNPointsExp(parseInt(e.target.value))} className="w-full" />
        </div>
        <div>
          <label className="text-gray-300 text-xs">Seed: {seed}</label>
          <input type="range" min={1} max={300} step={1} value={seed} onChange={(e) => setSeed(parseInt(e.target.value))} className="w-full" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <Plot
          data={[
            {
              x: result.xInside,
              y: result.yInside,
              type: 'scatter' as const,
              mode: 'markers' as const,
              name: 'accepted (inside)',
              marker: { color: 'rgba(34,197,94,0.45)', size: 3 },
            },
            {
              x: result.xOutside,
              y: result.yOutside,
              type: 'scatter' as const,
              mode: 'markers' as const,
              name: 'rejected (outside)',
              marker: { color: 'rgba(248,113,113,0.35)', size: 3 },
            },
          ]}
          layout={{
            title: { text: `First 2 coordinates (showing ${result.nPlot} points)` },
            xaxis: { title: { text: 'x1' }, range: [-1.1, 1.1], color: '#9ca3af', scaleanchor: 'y' },
            yaxis: { title: { text: 'x2' }, range: [-1.1, 1.1], color: '#9ca3af' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            legend: { bgcolor: 'rgba(0,0,0,0.3)' },
            height: 380,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
        <Plot
          data={[
            {
              x: result.dimAxis,
              y: result.theoryAxis,
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              line: { color: '#f472b6', dash: 'dash' },
              marker: { size: 6 },
              name: 'theoretical fraction',
            },
            {
              x: [nDims],
              y: [result.frac],
              type: 'scatter' as const,
              mode: 'markers' as const,
              marker: { color: '#22d3ee', symbol: 'star', size: 12 },
              name: 'MC estimate',
            },
          ]}
          layout={{
            title: { text: 'Accepted Fraction vs Dimension' },
            xaxis: { title: { text: 'dimension' }, color: '#9ca3af' },
            yaxis: { title: { text: 'fraction in unit sphere' }, type: 'log', color: '#9ca3af' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            legend: { bgcolor: 'rgba(0,0,0,0.3)' },
            height: 380,
            margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
        <div className="bg-[#0a0a15] rounded p-3">
          <div className="text-gray-500">Accepted / total</div>
          <div className="text-white font-mono">{result.insideCount} / {nPoints}</div>
        </div>
        <div className="bg-[#0a0a15] rounded p-3">
          <div className="text-gray-500">Acceptance fraction</div>
          <div className="text-white font-mono">{result.frac.toFixed(6)}</div>
        </div>
        <div className="bg-[#0a0a15] rounded p-3">
          <div className="text-gray-500">Estimated sphere volume</div>
          <div className="text-white font-mono">{result.estVolume.toFixed(6)}</div>
        </div>
        {result.piEstimate !== null && (
          <div className="bg-[#0a0a15] rounded p-3">
            <div className="text-gray-500">Pi estimate</div>
            <div className="text-white font-mono">{result.piEstimate.toFixed(6)}</div>
          </div>
        )}
      </div>
    </div>
  );
}
