'use client';

import dynamic from 'next/dynamic';
import { useState, useMemo } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const L = 1;
const Nx = 100;
const dx = L / (Nx - 1);
const dt = 0.00001;

const initialConditions = {
  sine: (x: number) => Math.sin(Math.PI * x),
  gaussian: (x: number) => Math.exp( - (((x - 0.5) / 0.1) ** 2) ),
  step: (x: number) => x < 0.5 ? 1 : 0,
};

function solveHeat1D(initialFunc: (x: number) => number, alpha: number, t: number) {
  const steps = Math.floor(t / dt);
  let u = Array.from({ length: Nx }, (_, i) => initialFunc(i * dx));
  for (let n = 0; n < steps; n++) {
    const uNew = [...u];
    for (let i = 1; i < Nx - 1; i++) {
      uNew[i] = u[i] + alpha * dt / (dx * dx) * (u[i - 1] - 2 * u[i] + u[i + 1]);
    }
    u = uNew;
  }
  return u;
}

export default function PDEs() {
  const [alpha, setAlpha] = useState(1);
  const [t, setT] = useState(0);
  const [initialType, setInitialType] = useState<'sine' | 'gaussian' | 'step'>('sine');

  const u = useMemo(() => solveHeat1D(initialConditions[initialType], alpha, t), [initialType, alpha, t]);

  const x = Array.from({ length: Nx }, (_, i) => i * dx);

  const data = [
    {
      x,
      y: u,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: `u(x, t=${t.toFixed(3)})`,
      line: { color: 'blue' },
    },
  ];

  const layout = {
    title: { text: '1D Heat Equation Solution' },
    xaxis: { title: { text: 'Position x' }, range: [0, L] },
    yaxis: { title: { text: 'Temperature u(x,t)' } },
    showlegend: true,
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">Partial Differential Equations: 1D Heat Equation</h1>
      <p className="mb-4">
        Interactive visualization of the 1D heat equation ∂u/∂t = α ∂²u/∂x².
        Adjust parameters below to see how the solution evolves.
      </p>

      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Initial Condition</label>
        <select
          value={initialType}
          onChange={(e) => setInitialType(e.target.value as 'sine' | 'gaussian' | 'step')}
          className="border rounded p-2"
        >
          <option value="sine">Sine Wave</option>
          <option value="gaussian">Gaussian</option>
          <option value="step">Step Function</option>
        </select>
      </div>

      <div className="mb-4">
        <label htmlFor="alpha-slider" className="block text-sm font-medium mb-2">
          Diffusivity α: {alpha.toFixed(2)}
        </label>
        <input
          id="alpha-slider"
          type="range"
          min="0.01"
          max="2"
          step="0.01"
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          className="w-full"
        />
      </div>

      <div className="mb-4">
        <label htmlFor="t-slider" className="block text-sm font-medium mb-2">
          Time t: {t.toFixed(3)}
        </label>
        <input
          id="t-slider"
          type="range"
          min="0"
          max="1"
          step="0.001"
          value={t}
          onChange={(e) => setT(Number(e.target.value))}
          className="w-full"
        />
      </div>

      <div className="mb-6">
        <Plot
          data={data}
          layout={layout}
          style={{ width: '100%', height: '500px' }}
        />
      </div>

      <div className="mt-6">
        <h2 className="text-2xl font-semibold mb-2">Explanation</h2>
        <p>
          The 1D heat equation models heat diffusion in a rod. The solution u(x,t) represents the temperature at position x and time t.
          Boundary conditions are assumed to be u(0,t)=u(1,t)=0 (insulated ends).
        </p>
        <h2 className="text-2xl font-semibold mb-2 mt-4">References</h2>
        <ul className="list-disc list-inside">
          <li>Wikipedia: Heat Equation</li>
          <li>Textbooks: Partial Differential Equations by Strauss, etc.</li>
        </ul>
      </div>
    </div>
  );
}