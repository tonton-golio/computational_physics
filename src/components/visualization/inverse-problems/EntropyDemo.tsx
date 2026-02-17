'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

function gaussPdf(x: number, mu: number, sigma: number): number {
  return (1 / (sigma * Math.sqrt(2 * Math.PI))) *
    Math.exp(-0.5 * ((x - mu) / sigma) ** 2);
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

export default function EntropyDemo({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [loc, setLoc] = useState(-8);
  const [scale, setScale] = useState(3.0);
  const [sizeExp, setSizeExp] = useState(8);
  const size = Math.pow(2, sizeExp);

  const result = useMemo(() => {
    const rng = seededRandom(42);

    // Discrete: sample from normal, round to integers
    const xDiscrete: number[] = [];
    for (let i = 0; i < size; i++) {
      xDiscrete.push(Math.round(boxMullerNormal(rng, loc, scale)));
    }

    // Compute discrete entropy
    const counts: Record<number, number> = {};
    for (const v of xDiscrete) {
      counts[v] = (counts[v] || 0) + 1;
    }
    let H_discrete = 0;
    for (const key of Object.keys(counts)) {
      const p = counts[Number(key)] / xDiscrete.length;
      if (p > 0) {
        H_discrete -= p * Math.log2(p);
      }
    }

    // Continuous: evaluate Gaussian PDF on linspace and compute differential entropy
    const nPts = 500;
    const xCont: number[] = [];
    const fxCont: number[] = [];
    const xMin = -40;
    const xMax = 40;
    const dx = (xMax - xMin) / (nPts - 1);
    for (let i = 0; i < nPts; i++) {
      const x = xMin + i * dx;
      xCont.push(x);
      fxCont.push(gaussPdf(x, loc, scale));
    }

    let H_continuous = 0;
    for (let i = 0; i < nPts; i++) {
      if (fxCont[i] > 1e-15) {
        H_continuous -= fxCont[i] * Math.log2(fxCont[i]) * dx;
      }
    }

    // Build histogram bins for discrete
    const minVal = Math.min(...xDiscrete);
    const maxVal = Math.max(...xDiscrete);
    const bins: number[] = [];
    const binCounts: number[] = [];
    for (let v = minVal; v <= maxVal; v++) {
      bins.push(v);
      binCounts.push(counts[v] || 0);
    }

    return {
      bins,
      binCounts,
      H_discrete,
      xCont,
      fxCont,
      H_continuous,
    };
  }, [loc, scale, size]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Discrete vs Continuous Entropy</h3>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-gray-300 text-sm">Location (mean): {loc}</label>
          <input
            type="range" min={-20} max={4} step={1} value={loc}
            onChange={(e) => setLoc(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-gray-300 text-sm">Scale (std): {scale.toFixed(1)}</label>
          <input
            type="range" min={0.5} max={10} step={0.1} value={scale}
            onChange={(e) => setScale(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-gray-300 text-sm">Sample size: 2^{sizeExp} = {size}</label>
          <input
            type="range" min={4} max={10} step={1} value={sizeExp}
            onChange={(e) => setSizeExp(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <Plot
            data={[{
              x: result.bins,
              y: result.binCounts,
              type: 'bar' as const,
              marker: { color: 'rgba(239,68,68,0.8)' },
              name: 'Discrete samples',
            }]}
            layout={{
              title: { text: `Discrete Entropy: H_s = ${result.H_discrete.toFixed(3)} bits` },
              xaxis: { title: { text: 'Value' }, range: [-40, 40], color: '#9ca3af' },
              yaxis: { title: { text: 'Count' }, color: '#9ca3af' },
              height: 320,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#9ca3af' },
              margin: { t: 40, b: 50, l: 50, r: 20 },
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
        <div>
          <Plot
            data={[{
              x: result.xCont,
              y: result.fxCont,
              type: 'scatter' as const,
              mode: 'lines' as const,
              fill: 'tozeroy',
              fillcolor: 'rgba(147,51,234,0.5)',
              line: { color: 'rgba(147,51,234,1)' },
              name: 'Gaussian PDF',
            }]}
            layout={{
              title: { text: `Differential Entropy: H_d = ${result.H_continuous.toFixed(3)} bits` },
              xaxis: { title: { text: 'Value' }, range: [-40, 40], color: '#9ca3af' },
              yaxis: { title: { text: 'f(x)' }, color: '#9ca3af' },
              height: 320,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#9ca3af' },
              margin: { t: 40, b: 50, l: 50, r: 20 },
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
      <p className="text-gray-500 text-xs mt-3">
        Discrete Shannon entropy is always non-negative. Differential entropy can be negative and is not a direct measure of information content. Note that differential entropy is translation-invariant.
      </p>
    </div>
  );
}
