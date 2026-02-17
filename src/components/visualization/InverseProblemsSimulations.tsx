'use client';

import React, { useState } from 'react';
import Plotly from 'react-plotly.js';

interface SimulationProps {
  id: string;
}

// Lotka-Volterra Parameter Estimation Simulation
function LotkaVolterraEstimationSim({ }: SimulationProps) {
  // True parameters for simulated data
  const TRUE_ALPHA = 1.5;
  const TRUE_BETA = 0.1;
  const TRUE_GAMMA = 1.5;
  const TRUE_DELTA = 0.075;

  // Initial conditions
  const X0 = 10;
  const Y0 = 5;
  const T0 = 0;
  const TF = 50;
  const DT = 0.1;

  function solveLotkaVolterra(alpha: number, beta: number, gamma: number, delta: number, x0: number, y0: number, t0: number, tf: number, dt: number) {
    let t = t0;
    let x = x0;
    let y = y0;
    const times: number[] = [];
    const xs: number[] = [];
    const ys: number[] = [];
    while (t <= tf + dt) { // +dt to include tf
      times.push(t);
      xs.push(x);
      ys.push(y);
      const dxdt = alpha * x - beta * x * y;
      const dydt = -gamma * y + delta * x * y;
      x += dxdt * dt;
      y += dydt * dt;
      t += dt;
    }
    return { times, xs, ys };
  }

  // Simulate data with true parameters
  const trueData = solveLotkaVolterra(TRUE_ALPHA, TRUE_BETA, TRUE_GAMMA, TRUE_DELTA, X0, Y0, T0, TF, DT);

  const [alpha, setAlpha] = useState(1.0);
  const [beta, setBeta] = useState(0.1);
  const [gamma, setGamma] = useState(1.0);
  const [delta, setDelta] = useState(0.05);

  // Fit data with current parameters
  const fitData = solveLotkaVolterra(alpha, beta, gamma, delta, X0, Y0, T0, TF, DT);

  // Calculate residuals
  const residualsX = trueData.xs.map((trueX, i) => trueX - fitData.xs[i]);
  const residualsY = trueData.ys.map((trueY, i) => trueY - fitData.ys[i]);

  const plotData = [
    {
      x: trueData.times,
      y: trueData.xs,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'True Prey Population',
      line: { color: 'blue' }
    },
    {
      x: trueData.times,
      y: trueData.ys,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'True Predator Population',
      line: { color: 'red' }
    },
    {
      x: fitData.times,
      y: fitData.xs,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Fitted Prey',
      line: { color: 'blue', dash: 'dash' }
    },
    {
      x: fitData.times,
      y: fitData.ys,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Fitted Predator',
      line: { color: 'red', dash: 'dash' }
    }
  ];

  const residualData = [
    {
      x: trueData.times,
      y: residualsX,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Prey Residuals',
      line: { color: 'blue' }
    },
    {
      x: trueData.times,
      y: residualsY,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Predator Residuals',
      line: { color: 'red' }
    }
  ];

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Lotka-Volterra Parameter Estimation</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-white">α (Prey Growth): {alpha.toFixed(2)}</label>
          <input
            type="range"
            min={0.5}
            max={3}
            step={0.1}
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">β (Predation Rate): {beta.toFixed(3)}</label>
          <input
            type="range"
            min={0.01}
            max={0.3}
            step={0.01}
            value={beta}
            onChange={(e) => setBeta(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">γ (Predator Death): {gamma.toFixed(2)}</label>
          <input
            type="range"
            min={0.5}
            max={3}
            step={0.1}
            value={gamma}
            onChange={(e) => setGamma(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white">δ (Predator Efficiency): {delta.toFixed(3)}</label>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.01}
            value={delta}
            onChange={(e) => setDelta(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h4 className="text-lg font-semibold mb-2 text-white">Population Dynamics</h4>
          <Plotly
            data={plotData}
            layout={{
              title: 'Lotka-Volterra Model Fit',
              xaxis: { title: 'Time' },
              yaxis: { title: 'Population' },
              height: 400,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#9ca3af' }
            }}
            config={{ displayModeBar: false }}
          />
        </div>
        <div>
          <h4 className="text-lg font-semibold mb-2 text-white">Residuals</h4>
          <Plotly
            data={residualData}
            layout={{
              title: 'Fit Residuals',
              xaxis: { title: 'Time' },
              yaxis: { title: 'Residual' },
              height: 400,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#9ca3af' }
            }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>

      <div className="mt-4 text-sm text-gray-300">
        <p>Adjust parameters to minimize residuals. True values: α=1.5, β=0.1, γ=1.5, δ=0.075</p>
      </div>
    </div>
  );
}

export const INVERSE_PROBLEMS_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'lotka-volterra-estimation': LotkaVolterraEstimationSim,
};