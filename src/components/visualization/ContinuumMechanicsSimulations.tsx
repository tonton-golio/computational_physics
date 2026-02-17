'use client';

import React, { useState, useCallback, useMemo } from 'react';
import Plotly from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

interface SimulationProps {
  id: string;
}

// Stress-Strain Simulation
function StressStrainSim({ id }: SimulationProps) {
  const [params, setParams] = useState({
    E: 200, // Young's modulus, GPa
    alpha: 0.002,
    n: 5,
    maxStrain: 0.05, // 5%
    numPoints: 100
  });

  // Bisection method to solve for sigma given epsilon
  const solveSigma = useCallback((epsilon: number, E: number, alpha: number, n: number): number => {
    if (epsilon === 0) return 0;
    let low = 0;
    let high = E * epsilon * 10; // upper bound
    let mid = 0;
    for (let i = 0; i < 100; i++) {
      mid = (low + high) / 2;
      const elastic = mid / E;
      const plastic = alpha * Math.pow(mid / E, n);
      const calcEpsilon = elastic + plastic;
      if (calcEpsilon < epsilon) {
        low = mid;
      } else {
        high = mid;
      }
    }
    return mid;
  }, []);

  const generateData = useMemo(() => {
    const { E, alpha, n, maxStrain, numPoints } = params;
    const epsilonValues: number[] = [];
    const linearSigma: number[] = [];
    const nonlinearSigma: number[] = [];

    for (let i = 0; i <= numPoints; i++) {
      const epsilon = (i / numPoints) * maxStrain;
      epsilonValues.push(epsilon);
      linearSigma.push(E * epsilon);
      nonlinearSigma.push(solveSigma(epsilon, E, alpha, n));
    }

    const data: Data[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x: epsilonValues,
        y: linearSigma,
        name: 'Linear (Hooke\'s Law)',
        line: { color: '#3b82f6' }
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: epsilonValues,
        y: nonlinearSigma,
        name: 'Nonlinear (Ramberg-Osgood)',
        line: { color: '#ef4444' }
      }
    ];

    const layout: Partial<Layout> = {
      title: 'Stress-Strain Curve',
      xaxis: { title: 'Strain (ε)' },
      yaxis: { title: 'Stress (σ) [GPa]' },
      height: 500,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(15,15,25,1)',
      font: { color: '#9ca3af' }
    };

    return { data, layout };
  }, [params, solveSigma]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Stress-Strain Simulation</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-white">Young's Modulus (E) [GPa]: {params.E}</label>
          <input
            type="range"
            min={50}
            max={500}
            step={10}
            value={params.E}
            onChange={(e) => setParams(p => ({ ...p, E: Number(e.target.value) }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Hardening Coefficient (α): {params.alpha.toFixed(3)}</label>
          <input
            type="range"
            min={0.001}
            max={0.01}
            step={0.001}
            value={params.alpha}
            onChange={(e) => setParams(p => ({ ...p, alpha: Number(e.target.value) }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Hardening Exponent (n): {params.n}</label>
          <input
            type="range"
            min={1}
            max={10}
            step={0.5}
            value={params.n}
            onChange={(e) => setParams(p => ({ ...p, n: Number(e.target.value) }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">Max Strain: {params.maxStrain.toFixed(3)}</label>
          <input
            type="range"
            min={0.01}
            max={0.1}
            step={0.005}
            value={params.maxStrain}
            onChange={(e) => setParams(p => ({ ...p, maxStrain: Number(e.target.value) }))}
            className="w-full"
          />
        </div>
      </div>
      <Plotly
        data={generateData.data}
        layout={generateData.layout}
        config={{ displayModeBar: false }}
      />
      <div className="mt-4 text-sm text-gray-300">
        <p>The linear curve follows Hooke's law: σ = Eε</p>
        <p>The nonlinear curve uses the Ramberg-Osgood model: ε = σ/E + α(σ/E)^n</p>
      </div>
    </div>
  );
}

export const CONTINUUM_MECHANICS_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'stress-strain-sim': StressStrainSim,
};