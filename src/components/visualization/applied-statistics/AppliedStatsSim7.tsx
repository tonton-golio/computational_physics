'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

/**
 * PDF Visualization demo.
 * Shows samples from Uniform, Poisson (approximated), Binomial, and Gaussian distributions
 * side by side, controlled by a single sample-size slider.
 */
export default function AppliedStatsSim7({ }: SimulationProps) {
  const [size, setSize] = useState(500);
  const [nBinomial, setNBinomial] = useState(100);
  const [pBinomial, setPBinomial] = useState(0.2);
  const [seed, setSeed] = useState(69);

  const result = useMemo(() => {
    // Seeded pseudo-random
    let s = seed;
    const seededRandom = () => {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };
    const seededNormal = () => {
      const u1 = seededRandom();
      const u2 = seededRandom();
      return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    };

    // Uniform [0, 1]
    const uniform: number[] = [];
    for (let i = 0; i < size; i++) {
      uniform.push(seededRandom());
    }

    // Binomial: sum of n Bernoulli trials
    const binomial: number[] = [];
    for (let i = 0; i < size; i++) {
      let successes = 0;
      for (let j = 0; j < nBinomial; j++) {
        if (seededRandom() < pBinomial) successes++;
      }
      binomial.push(successes);
    }

    // Poisson approximation using inverse transform
    // For lambda = n*p
    const lambda = nBinomial * pBinomial;
    const poisson: number[] = [];
    for (let i = 0; i < size; i++) {
      const L = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= seededRandom();
      } while (p > L);
      poisson.push(k - 1);
    }

    // Gaussian
    const gaussian: number[] = [];
    for (let i = 0; i < size; i++) {
      gaussian.push(seededNormal());
    }

    return { uniform, binomial, poisson, gaussian, lambda };
  }, [size, nBinomial, pBinomial, seed]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Probability Density Functions</h3>
      <p className="text-sm text-gray-300 mb-4">
        Compare samples drawn from four fundamental distributions: Uniform, Poisson, Binomial, and Gaussian.
        Adjust the number of samples and binomial parameters to see how the histograms change.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="text-sm text-gray-400">Samples: {size}</label>
          <input type="range" min={50} max={5000} step={50} value={size}
            onChange={e => setSize(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-gray-400">Binomial n: {nBinomial}</label>
          <input type="range" min={5} max={200} step={5} value={nBinomial}
            onChange={e => setNBinomial(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-gray-400">Binomial p: {pBinomial.toFixed(2)}</label>
          <input type="range" min={0.01} max={0.99} step={0.01} value={pBinomial}
            onChange={e => setPBinomial(+e.target.value)} className="w-full" />
        </div>
        <div>
          <label className="text-sm text-gray-400">Seed: {seed}</label>
          <input type="range" min={1} max={200} step={1} value={seed}
            onChange={e => setSeed(+e.target.value)} className="w-full" />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Plot
          data={[{
            x: result.uniform,
            type: 'histogram',
            nbinsx: 40,
            marker: { color: 'rgba(249,115,22,0.7)' },
            name: 'Uniform',
          }]}
          layout={{
            title: { text: 'Uniform', font: { color: '#fff', size: 13 } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af', size: 10 },
            margin: { t: 35, r: 10, b: 30, l: 35 },
            xaxis: { gridcolor: '#1e1e2e' },
            yaxis: { gridcolor: '#1e1e2e' },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 280 }}
        />
        <Plot
          data={[{
            x: result.poisson,
            type: 'histogram',
            nbinsx: 40,
            marker: { color: 'rgba(236,72,153,0.7)' },
            name: 'Poisson',
          }]}
          layout={{
            title: { text: `Poisson (lambda=${result.lambda.toFixed(1)})`, font: { color: '#fff', size: 13 } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af', size: 10 },
            margin: { t: 35, r: 10, b: 30, l: 35 },
            xaxis: { gridcolor: '#1e1e2e' },
            yaxis: { gridcolor: '#1e1e2e' },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 280 }}
        />
        <Plot
          data={[{
            x: result.binomial,
            type: 'histogram',
            nbinsx: 40,
            marker: { color: 'rgba(249,115,22,0.7)' },
            name: 'Binomial',
          }]}
          layout={{
            title: { text: `Binomial (n=${nBinomial}, p=${pBinomial.toFixed(2)})`, font: { color: '#fff', size: 13 } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af', size: 10 },
            margin: { t: 35, r: 10, b: 30, l: 35 },
            xaxis: { gridcolor: '#1e1e2e' },
            yaxis: { gridcolor: '#1e1e2e' },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 280 }}
        />
        <Plot
          data={[{
            x: result.gaussian,
            type: 'histogram',
            nbinsx: 30,
            marker: { color: 'rgba(236,72,153,0.7)' },
            name: 'Gaussian',
          }]}
          layout={{
            title: { text: 'Gaussian', font: { color: '#fff', size: 13 } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af', size: 10 },
            margin: { t: 35, r: 10, b: 30, l: 35 },
            xaxis: { gridcolor: '#1e1e2e' },
            yaxis: { gridcolor: '#1e1e2e' },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 280 }}
        />
      </div>
    </div>
  );
}
