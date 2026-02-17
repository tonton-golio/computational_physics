"use client";
import React, { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

interface HistoryPoint {
  t: number;
  x: number;
  y: number;
}

const GeneNetsPage: React.FC = () => {
  const [params, setParams] = useState({
    alpha: 1.5,
    beta: 1.0,
    delta: 1.0,
    gamma: 3.0,
    x0: 1.0,
    y0: 1.0,
    tMax: 50,
    dt: 0.05,
    gridSize: 15
  });
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const fullHistory = useMemo(() => {
    const { alpha, beta, delta, gamma, x0, y0, tMax, dt } = params;
    let x = x0;
    let y = y0;
    let t = 0;
    const history: HistoryPoint[] = [{ t, x, y }];
    while (t < tMax) {
      const dxdt = alpha * x - beta * x * y;
      const dydt = delta * x * y - gamma * y;
      x = Math.max(0, x + dxdt * dt);
      y = Math.max(0, y + dydt * dt);
      t += dt;
      history.push({ t, x, y });
    }
    return history;
  }, [params]);

  const togglePlay = () => {
    if (currentIndex >= fullHistory.length - 1) {
      setPlaying(false);
    } else {
      setPlaying(!playing);
    }
  };

  const reset = () => {
    setPlaying(false);
    setCurrentIndex(0);
  };

  useEffect(() => {
    let frameId: number;
    if (playing && currentIndex < fullHistory.length - 1) {
      frameId = requestAnimationFrame(() => {
        setCurrentIndex((prev) => Math.min(prev + 10, fullHistory.length - 1)); // speed
      });
    }
    return () => cancelAnimationFrame(frameId);
  }, [playing, currentIndex, fullHistory.length]);

  // Phase portrait traces
  const alphaOverBeta = params.alpha / params.beta;
  const gammaOverDelta = params.gamma / params.delta;
  const maxPop = 10;

  const preyNullclineX = Array.from({length: 100}, (_, i) => (i/99) * maxPop);
  const preyNullclineY = new Array(100).fill(alphaOverBeta);

  const predNullclineX = new Array(100).fill(gammaOverDelta);
  const predNullclineY = Array.from({length: 100}, (_, i) => (i/99) * maxPop);

  const trajX = fullHistory.slice(0, currentIndex + 1).map(p => p.x);
  const trajY = fullHistory.slice(0, currentIndex + 1).map(p => p.y);

  const quiverData = (() => {
    const { alpha, beta, delta, gamma, gridSize } = params;
    const xmin = 0, xmax = maxPop, ymin = 0, ymax = maxPop;
    const quiverX: number[] = [];
    const quiverY: number[] = [];
    const quiverU: number[] = [];
    const quiverV: number[] = [];
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const px = xmin + (i / (gridSize - 1)) * (xmax - xmin);
        const py = ymin + (j / (gridSize - 1)) * (ymax - ymin);
        const dxdt = alpha * px - beta * px * py;
        const dydt = delta * px * py - gamma * py;
        const mag = Math.sqrt(dxdt**2 + dydt**2) || 0.1;
        quiverX.push(px);
        quiverY.push(py);
        quiverU.push((dxdt / mag) * 0.3);
        quiverV.push((dydt / mag) * 0.3);
      }
    }
    return [{
      type: 'scatter',
      x: quiverX,
      y: quiverY,
      u: quiverU,
      v: quiverV,
      mode: 'markers',
      marker: {
        color: 'rgba(0,100,200,0.8)',
        size: 8,
        symbol: 'arrow-bar-up'
      },
      hoverinfo: 'skip',
      name: 'Vector Field'
    }];
  })();

  const phaseData: Data[] = [
    {
      type: 'scatter',
      x: preyNullclineX,
      y: preyNullclineY,
      mode: 'lines',
      line: { color: 'green', dash: 'dash' },
      name: 'Prey nullcline (dx/dt=0)'
    },
    {
      type: 'scatter',
      x: predNullclineX,
      y: predNullclineY,
      mode: 'lines',
      line: { color: 'red', dash: 'dash' },
      name: 'Predator nullcline (dy/dt=0)'
    },
    {
      type: 'scatter',
      x: trajX,
      y: trajY,
      mode: 'lines',
      line: { color: 'blue', width: 3 },
      name: 'Trajectory'
    },
    ...(quiverData as Data[])
  ];

  const phaseLayout: Partial<Layout> = {
    title: { text: 'Lotka-Volterra Phase Portrait' },
    xaxis: { title: { text: 'Prey (x)' }, range: [0, maxPop] },
    yaxis: { title: { text: 'Predator (y)' }, range: [0, maxPop] },
    showlegend: true,
    width: 600,
    height: 500
  };

  // Time series
  const timeX = fullHistory.slice(0, currentIndex + 1).map(p => p.t);
  const preyY = fullHistory.slice(0, currentIndex + 1).map(p => p.x);
  const predY = fullHistory.slice(0, currentIndex + 1).map(p => p.y);

  const timeData: Data[] = [
    {
      type: 'scatter',
      x: timeX,
      y: preyY,
      mode: 'lines',
      name: 'Prey',
      line: { color: 'green' }
    },
    {
      type: 'scatter',
      x: timeX,
      y: predY,
      mode: 'lines',
      name: 'Predator',
      yaxis: 'y2',
      line: { color: 'red' }
    }
  ];

  const timeLayout: Partial<Layout> = {
    title: { text: 'Populations over Time' },
    xaxis: { title: { text: 'Time' } },
    yaxis: { title: { text: 'Prey' }, side: 'left' },
    yaxis2: {
      title: { text: 'Predator' },
      side: 'right',
      overlaying: 'y'
    },
    width: 600,
    height: 500
  };

  const updateParam = (key: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams({ ...params, [key]: parseFloat(e.target.value) });
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">Dynamical Models in Gene Regulatory Networks</h1>
      <p className="mb-8 text-lg">
        Gene regulatory networks (GRNs) model how genes interact to control expression. We use Boolean networks for discrete on/off states and ODEs for continuous dynamics.
      </p>
      <p className="mb-8">
        Boolean: genes 0/1, logical rules. ODE: rates with Hill functions.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div>
          <Plot data={phaseData} layout={phaseLayout} config={{displayModeBar: true}} />
        </div>
        <div>
          <Plot data={timeData} layout={timeLayout} config={{displayModeBar: true}} />
        </div>
      </div>
      <div className="bg-gray-100 p-6 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">Lotka-Volterra Predator-Prey Model (Example)</h2>
        <p className="mb-4">
          dx/dt = αx - βxy<br />
          dy/dt = δxy - γy
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-4">
          <div>
            <label className="block text-sm">α (prey growth): {params.alpha.toFixed(2)}</label>
            <input type="range" min={0.1} max={3} step={0.1} value={params.alpha} onChange={updateParam('alpha')} className="w-full" />
          </div>
          <div>
            <label className="block text-sm">β (predation): {params.beta.toFixed(2)}</label>
            <input type="range" min={0.1} max={2} step={0.1} value={params.beta} onChange={updateParam('beta')} className="w-full" />
          </div>
          <div>
            <label className="block text-sm">δ (conversion): {params.delta.toFixed(2)}</label>
            <input type="range" min={0.1} max={2} step={0.1} value={params.delta} onChange={updateParam('delta')} className="w-full" />
          </div>
          <div>
            <label className="block text-sm">γ (pred death): {params.gamma.toFixed(2)}</label>
            <input type="range" min={0.1} max={5} step={0.1} value={params.gamma} onChange={updateParam('gamma')} className="w-full" />
          </div>
          <div>
            <label className="block text-sm">x0: {params.x0.toFixed(2)}</label>
            <input type="range" min={0.1} max={5} step={0.1} value={params.x0} onChange={updateParam('x0')} className="w-full" />
          </div>
          <div>
            <label className="block text-sm">y0: {params.y0.toFixed(2)}</label>
            <input type="range" min={0.1} max={5} step={0.1} value={params.y0} onChange={updateParam('y0')} className="w-full" />
          </div>
        </div>
        <div className="flex gap-4">
          <button onClick={reset} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Reset</button>
          <button onClick={togglePlay} className={`px-4 py-2 ${playing ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'} text-white rounded`}>
            {playing ? 'Pause' : 'Play'}
          </button>
        </div>
      </div>
      <div className="mt-8 p-6 bg-yellow-50 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">Notes on GRNs</h2>
        <ul className="list-disc pl-6">
          <li><strong>Boolean Networks:</strong> Discrete, logical gates. Attractors = stable states.</li>
          <li><strong>ODE Models:</strong> Continuous rates, Hill functions for nonlinearity.</li>
          <li><strong>Bifurcations:</strong> Parameter changes lead to new behaviors (e.g., bistability in toggle switch).</li>
          <li>Phase portraits show fixed points, limit cycles.</li>
        </ul>
      </div>
    </div>
  );
};

export default GeneNetsPage;
