'use client';

import dynamic from 'next/dynamic';
import { useState } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function QuantumMechanics() {
  const [n, setN] = useState(1);

  // Constants (in units where ħ=1, m=1, L=1 for simplicity)
  const L = 1;
  const hbar = 1;
  const m = 1;

  // Energy calculation
  const E = (n ** 2 * Math.PI ** 2 * hbar ** 2) / (2 * m * L ** 2);

  // Generate x values
  const numPoints = 200;
  const x = Array.from({ length: numPoints }, (_, i) => (i / (numPoints - 1)) * L);

  // Wave function ψ_n(x)
  const psi = x.map(xi => Math.sqrt(2 / L) * Math.sin(n * Math.PI * xi / L));

  // Probability density |ψ_n(x)|²
  const prob = x.map(xi => (2 / L) * Math.sin(n * Math.PI * xi / L) ** 2);

  // Data for plots
  const dataPsi = [
    {
      x,
      y: psi,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: `Wave Function ψ_${n}(x)`,
      line: { color: 'blue' },
    },
  ];

  const dataProb = [
    {
      x,
      y: prob,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: `Probability Density |ψ_${n}(x)|²`,
      fill: 'tozeroy' as const,
      line: { color: 'red' },
    },
  ];

  const layoutPsi = {
    title: { text: 'Wave Function ψ_n(x)' },
    xaxis: { title: { text: 'Position x' }, range: [0, L] },
    yaxis: { title: { text: 'ψ_n(x)' } },
    showlegend: true,
  };

  const layoutProb = {
    title: { text: 'Probability Density |ψ_n(x)|²' },
    xaxis: { title: { text: 'Position x' }, range: [0, L] },
    yaxis: { title: { text: '|ψ_n(x)|²' } },
    showlegend: true,
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">Infinite Square Well Simulation</h1>
      <p className="mb-4">
        This simulation visualizes the wave functions and probability densities for the infinite square well potential.
        Adjust the quantum number n using the slider below.
      </p>

      <div className="mb-4">
        <label htmlFor="n-slider" className="block text-sm font-medium mb-2">
          Quantum number n: {n}
        </label>
        <input
          id="n-slider"
          type="range"
          min="1"
          max="10"
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
          className="w-full"
        />
      </div>

      <div className="mb-4">
        <p className="text-lg">
          Energy E_{n} = {E.toFixed(3)} (in units where ħ=1, m=1, L=1)
        </p>
        <p className="text-sm text-gray-600">
          Formula: E_n = (n² π² ħ²) / (2 m L²)
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <Plot
            data={dataPsi}
            layout={layoutPsi}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
        <div>
          <Plot
            data={dataProb}
            layout={layoutProb}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
      </div>

      <div className="mt-6">
        <h2 className="text-2xl font-semibold mb-2">References</h2>
        <ul className="list-disc list-inside">
          <li>MIT OpenCourseWare: Quantum Physics I (8.04) - Infinite Square Well</li>
          <li>PhET Interactive Simulations: Quantum Mechanics</li>
        </ul>
      </div>
    </div>
  );
}