'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

function boxMullerNormal(rng: () => number, mu: number, sigma: number): number {
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

export default function KLDivergence({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [loc1, setLoc1] = useState(5);
  const [scale1, setScale1] = useState(1.0);
  const [loc2, setLoc2] = useState(3);
  const [scale2, setScale2] = useState(1.0);

  const result = useMemo(() => {
    const rng1 = seededRandom(123);
    const rng2 = seededRandom(456);
    const sampleSize = 400;

    const x1: number[] = [];
    const x2: number[] = [];
    for (let i = 0; i < sampleSize; i++) {
      x1.push(boxMullerNormal(rng1, loc1, scale1));
      x2.push(boxMullerNormal(rng2, loc2, scale2));
    }

    // Build shared bins
    const allMin = Math.min(Math.min(...x1), Math.min(...x2));
    const allMax = Math.max(Math.max(...x1), Math.max(...x2));
    const nBins = 30;
    const binWidth = (allMax - allMin) / nBins;
    const binEdges: number[] = [];
    for (let i = 0; i <= nBins; i++) {
      binEdges.push(allMin + i * binWidth);
    }
    const binCenters = binEdges.slice(0, -1).map((e, i) => (e + binEdges[i + 1]) / 2);

    // Histogram counts
    const counts1 = new Array(nBins).fill(0);
    const counts2 = new Array(nBins).fill(0);
    for (const v of x1) {
      const idx = Math.min(Math.floor((v - allMin) / binWidth), nBins - 1);
      if (idx >= 0) counts1[idx]++;
    }
    for (const v of x2) {
      const idx = Math.min(Math.floor((v - allMin) / binWidth), nBins - 1);
      if (idx >= 0) counts2[idx]++;
    }

    // Normalize to probabilities
    const sum1 = counts1.reduce((a: number, b: number) => a + b, 0);
    const sum2 = counts2.reduce((a: number, b: number) => a + b, 0);
    const p1 = counts1.map((c: number) => c / sum1);
    const p2 = counts2.map((c: number) => c / sum2);

    // KL divergence D_KL(P || Q)
    const threshold = 0.01;
    let klDiv = 0;
    let method2 = false;
    for (let i = 0; i < nBins; i++) {
      if (p1[i] > threshold && p2[i] > threshold) {
        klDiv -= p1[i] * Math.log2(p2[i] / p1[i]);
      }
    }
    if (!isFinite(klDiv) || klDiv < 0) {
      // fallback: compute in reverse direction
      method2 = true;
      klDiv = 0;
      for (let i = 0; i < nBins; i++) {
        if (p2[i] > threshold && p1[i] > threshold) {
          klDiv += p1[i] * Math.log2(p1[i] / p2[i]);
        }
      }
    }

    return {
      binEdges,
      binCenters,
      counts1,
      counts2,
      klDiv,
      method2,
    };
  }, [loc1, scale1, loc2, scale2]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">KL-Divergence Visualization</h3>
      <p className="text-gray-400 text-sm mb-4">
        Adjust the parameters of two distributions (P and Q) to see how the KL-divergence changes.
        The KL-divergence measures how much distribution P diverges from distribution Q.
      </p>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="text-blue-400 text-sm">P mean: {loc1}</label>
          <input
            type="range" min={0} max={10} step={1} value={loc1}
            onChange={(e) => setLoc1(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-blue-400 text-sm">P std: {scale1.toFixed(1)}</label>
          <input
            type="range" min={0.2} max={5} step={0.1} value={scale1}
            onChange={(e) => setScale1(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-red-400 text-sm">Q mean: {loc2}</label>
          <input
            type="range" min={0} max={10} step={1} value={loc2}
            onChange={(e) => setLoc2(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-red-400 text-sm">Q std: {scale2.toFixed(1)}</label>
          <input
            type="range" min={0.2} max={5} step={0.1} value={scale2}
            onChange={(e) => setScale2(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <Plot
        data={[
          {
            x: result.binCenters,
            y: result.counts1,
            type: 'bar' as const,
            marker: { color: 'rgba(59,130,246,0.7)' },
            name: 'P (distribution 1)',
            width: result.binEdges[1] - result.binEdges[0],
          },
          {
            x: result.binCenters,
            y: result.counts2,
            type: 'bar' as const,
            marker: { color: 'rgba(239,68,68,0.5)' },
            name: 'Q (distribution 2)',
            width: result.binEdges[1] - result.binEdges[0],
          },
        ]}
        layout={{
          title: { text: `D_KL(P || Q) = ${result.klDiv.toFixed(4)} bits` },
          barmode: 'overlay',
          xaxis: { title: { text: 'Value' }, color: '#9ca3af' },
          yaxis: { title: { text: 'Count' }, color: '#9ca3af' },
          height: 400,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.3)' },
          margin: { t: 40, b: 50, l: 50, r: 20 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
      {result.method2 && (
        <p className="text-yellow-500 text-xs mt-2">Used alternate computation method due to zero-probability bins.</p>
      )}
      <p className="text-gray-500 text-xs mt-2">
        KL-divergence is non-negative and equals zero if and only if P and Q are identical.
        It is asymmetric: D_KL(P||Q) is generally not equal to D_KL(Q||P).
      </p>
    </div>
  );
}
