"use client";
import React, { useState, useCallback, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

const ContinuumMechanicsPage: React.FC = () => {
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
    let mid: number;
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
        x: epsilonValues,
        y: linearSigma,
        mode: 'lines',
        name: 'Hooke&apos;s Law (Linear)',
        line: { color: 'blue', width: 2 }
      },
      {
        type: 'scatter',
        x: epsilonValues,
        y: nonlinearSigma,
        mode: 'lines',
        name: 'Ramberg-Osgood (Nonlinear)',
        line: { color: 'red', width: 2 }
      }
    ];

    const layout: Partial<Layout> = {
      title: '1D Stress-Strain Relationship',
      xaxis: { title: 'Strain (ε)', range: [0, maxStrain] },
      yaxis: { title: 'Stress (σ) [GPa]' },
      showlegend: true,
      width: 800,
      height: 600
    };

    return { data, layout };
  }, [params, solveSigma]);

  const updateParam = (key: keyof typeof params) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams({ ...params, [key]: parseFloat(e.target.value) });
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">Continuum Mechanics: 1D Stress-Strain</h1>
      <p className="mb-8 text-lg">
        Explore Hooke&apos;s law and nonlinear elasticity using the Ramberg-Osgood model.
      </p>

      <div className="mb-8">
        <Plot data={generateData.data} layout={generateData.layout} config={{ displayModeBar: true }} />
      </div>

      <div className="bg-gray-100 p-6 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">Parameters</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <div>
            <label>{`E (Young's Modulus, GPa): ${params.E.toFixed(0)}`}</label>
            <input type="range" min={50} max={500} step={10} value={params.E} onChange={updateParam('E')} className="w-full" />
          </div>
          <div>
            <label>{`α: ${params.alpha.toFixed(4)}`}</label>
            <input type="range" min={0.0001} max={0.01} step={0.0001} value={params.alpha} onChange={updateParam('alpha')} className="w-full" />
          </div>
          <div>
            <label>{`n: ${params.n.toFixed(1)}`}</label>
            <input type="range" min={1} max={20} step={0.1} value={params.n} onChange={updateParam('n')} className="w-full" />
          </div>
          <div>
            <label>{`Max Strain: ${params.maxStrain.toFixed(3)}`}</label>
            <input type="range" min={0.01} max={0.1} step={0.005} value={params.maxStrain} onChange={updateParam('maxStrain')} className="w-full" />
          </div>
          <div>
            <label>{`Num Points: ${params.numPoints}`}</label>
            <input type="range" min={50} max={200} step={10} value={params.numPoints} onChange={updateParam('numPoints')} className="w-full" />
          </div>
        </div>
      </div>

      <div className="mt-8">
        <h2 className="text-2xl font-semibold mb-4">Equations</h2>
        <ul className="list-disc list-inside">
          <li><strong>Hooke's Law (Linear):</strong> σ = E ε</li>
          <li><strong>Ramberg-Osgood (Nonlinear):</strong> ε = σ/E + α (σ/E)<sup>n</sup></li>
        </ul>
        <p className="mt-4">
          The Ramberg-Osgood model captures nonlinear elastic behavior, often used in materials science for metals exhibiting power-law hardening.
        </p>
      </div>
    </div>
  );
};

export default ContinuumMechanicsPage;