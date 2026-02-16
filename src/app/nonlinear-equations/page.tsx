"use client";
import React, { useState, useCallback, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

interface Point {
  t?: number;
  x: number;
  y?: number;
}

const NonlinearEquationsPage: React.FC = () => {
  const [system, setSystem] = useState<'lotka' | 'logistic'>('lotka');
  const [params, setParams] = useState({
    alpha: 1.5, beta: 1.0, delta: 1.0, gamma: 3.0, // Lotka
    x0: 1.0, y0: 1.0, tMax: 50, dt: 0.05, gridSize: 15,
    r: 2.0, K: 1.0, // Logistic
    bifurcationRMin: 0.1, bifurcationRMax: 4.0
  });
  const [fullHistory, setFullHistory] = useState<Point[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const rafRef = useRef<number>(0);

  const integrate = useCallback(() => {
    const { x0, y0, tMax, dt } = params;
    let x = x0;
    let y = y0 || 0;
    let t = 0;
    const history: Point[] = [{ t, x, y }];
    while (t < tMax) {
      let dxdt: number, dydt = 0;
      if (system === 'lotka') {
        const { alpha, beta, delta, gamma } = params;
        dxdt = alpha * x - beta * x * y;
        dydt = delta * x * y - gamma * y;
      } else { // logistic
        const { r, K } = params;
        dxdt = r * x * (1 - x / K);
        dydt = 0; // 1D
      }
      x = Math.max(0, x + dxdt * dt);
      y = Math.max(0, y + dydt * dt);
      t += dt;
      history.push({ t, x, y });
    }
    setFullHistory(history);
    setCurrentIndex(0);
    setPlaying(false);
  }, [params, system]);

  useEffect(() => {
    integrate();
  }, [integrate]);

  useEffect(() => {
    let frameId: number;
    if (playing && currentIndex < fullHistory.length - 1) {
      frameId = requestAnimationFrame(() => {
        setCurrentIndex((prev) => Math.min(prev + 10, fullHistory.length - 1));
      });
    } else if (playing) {
      setPlaying(false);
    }
    return () => cancelAnimationFrame(frameId);
  }, [playing, currentIndex, fullHistory.length]);

  const generatePhaseData = () => {
    if (system === 'lotka') {
      const { alpha, beta, delta, gamma, gridSize } = params;
      const alphaOverBeta = alpha / beta;
      const gammaOverDelta = gamma / delta;
      const maxPop = 10;

      const preyNullclineX = Array.from({length: 100}, (_, i) => (i/99) * maxPop);
      const preyNullclineY = new Array(100).fill(alphaOverBeta);

      const predNullclineX = new Array(100).fill(gammaOverDelta);
      const predNullclineY = Array.from({length: 100}, (_, i) => (i/99) * maxPop);

      const trajX = fullHistory.slice(0, currentIndex + 1).map(p => p.x);
      const trajY = fullHistory.slice(0, currentIndex + 1).map(p => p.y);

      const quiverX: number[] = [];
      const quiverY: number[] = [];
      const quiverU: number[] = [];
      const quiverV: number[] = [];
      const xmin = 0, xmax = maxPop, ymin = 0, ymax = maxPop;
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
      const quiverData = {
        type: 'scatter',
        x: quiverX,
        y: quiverY,
        u: quiverU,
        v: quiverV,
        mode: 'markers',
        marker: { color: 'rgba(0,100,200,0.8)', size: 8, symbol: 'arrow-bar-up' },
        hoverinfo: 'skip',
        name: 'Vector Field'
      };

      return {
        data: [
          { type: 'scatter', x: preyNullclineX, y: preyNullclineY, mode: 'lines', line: { color: 'green', dash: 'dash' }, name: 'Prey nullcline' },
          { type: 'scatter', x: predNullclineX, y: predNullclineY, mode: 'lines', line: { color: 'red', dash: 'dash' }, name: 'Predator nullcline' },
          { type: 'scatter', x: trajX, y: trajY, mode: 'lines', line: { color: 'blue', width: 3 }, name: 'Trajectory' },
          quiverData
        ] as Data[],
        layout: {
          title: 'Lotka-Volterra Phase Portrait',
          xaxis: { title: 'Prey (x)', range: [0, maxPop] },
          yaxis: { title: 'Predator (y)', range: [0, maxPop] },
          showlegend: true,
          width: 600, height: 500
        } as Partial<Layout>
      };
    } else { // logistic - 1D phase is not standard, but plot x vs dx/dt or something, or just time series
      return {
        data: [] as Data[],
        layout: { title: 'Logistic: 1D System - See Time Series' } as Partial<Layout>
      };
    }
  };

  const { data: phaseData, layout: phaseLayout } = generatePhaseData();

  const generateTimeData = () => {
    const validHistory = fullHistory.slice(0, currentIndex + 1).filter(p => p.x !== undefined && p.y !== undefined);
    const timeX = validHistory.map(p => p.t);
    const xY = validHistory.map(p => p.x);
    const yY = validHistory.map(p => p.y);
    const data: Data[] = [
      { type: 'scatter', x: timeX, y: xY, mode: 'lines', name: 'x', line: { color: 'blue' } }
    ];
    if (system === 'lotka') {
      data.push({ type: 'scatter', x: timeX, y: yY, mode: 'lines', name: 'y', line: { color: 'red' }, yaxis: 'y2' });
    }
    return {
      data,
      layout: {
        title: 'Time Series',
        xaxis: { title: 'Time' },
        yaxis: { title: 'x' },
        ...(system === 'lotka' ? {
          yaxis2: { title: 'y', side: 'right', overlaying: 'y' }
        } : {}),
        width: 600, height: 500
      } as Partial<Layout>
    };
  };

  const { data: timeData, layout: timeLayout } = generateTimeData();

  const generateBifurcationData = () => {
    if (system === 'logistic') {
      const { bifurcationRMin, bifurcationRMax, K } = params;
      const rValues: number[] = [];
      const xFixed: number[] = [];
      for (let r = bifurcationRMin; r <= bifurcationRMax; r += 0.01) {
        rValues.push(r);
        // Fixed points: x=0, and x = K(1 - 1/r) if r>1
        xFixed.push(0);
        if (r > 1) {
          xFixed.push(K * (1 - 1/r));
        } else {
          xFixed.push(NaN); // to break the line
        }
      }
      return {
        data: [{ type: 'scatter', x: rValues, y: xFixed, mode: 'lines', line: { color: 'purple' }, name: 'Fixed Points' }] as Data[],
        layout: {
          title: 'Bifurcation Diagram: r vs Fixed Points',
          xaxis: { title: 'r' },
          yaxis: { title: 'x*' },
          width: 600, height: 500
        } as Partial<Layout>
      };
    } else {
      return { data: [] as Data[], layout: {} as Partial<Layout> };
    }
  };

  const { data: bifurcationData, layout: bifurcationLayout } = generateBifurcationData();

  const updateParam = (key: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setParams({ ...params, [key]: parseFloat(e.target.value) });
  };

  const togglePlay = () => setPlaying(!playing);
  const reset = () => { setPlaying(false); setCurrentIndex(0); };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">Nonlinear Differential Equations</h1>
      <p className="mb-8 text-lg">
        Explore phase portraits, vector fields, and bifurcations in nonlinear systems.
      </p>

      <div className="mb-4">
        <label className="mr-4">System:</label>
        <select value={system} onChange={(e) => setSystem(e.target.value)} className="border p-2">
          <option value="lotka">Lotka-Volterra</option>
          <option value="logistic">Logistic</option>
        </select>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div>
          <Plot data={phaseData} layout={phaseLayout} config={{displayModeBar: true}} />
        </div>
        <div>
          <Plot data={timeData} layout={timeLayout} config={{displayModeBar: true}} />
        </div>
      </div>

      {system === 'logistic' && (
        <div className="mb-8">
          <Plot data={bifurcationData} layout={bifurcationLayout} config={{displayModeBar: true}} />
        </div>
      )}

      <div className="bg-gray-100 p-6 rounded-lg">
        <h2 className="text-2xl font-semibold mb-4">Parameters</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-4">
          {system === 'lotka' ? (
            <>
              <div>
                <label>α: {params.alpha.toFixed(2)}</label>
                <input type="range" min={0.1} max={3} step={0.1} value={params.alpha} onChange={updateParam('alpha')} className="w-full" />
              </div>
              <div>
                <label>β: {params.beta.toFixed(2)}</label>
                <input type="range" min={0.1} max={2} step={0.1} value={params.beta} onChange={updateParam('beta')} className="w-full" />
              </div>
              <div>
                <label>δ: {params.delta.toFixed(2)}</label>
                <input type="range" min={0.1} max={2} step={0.1} value={params.delta} onChange={updateParam('delta')} className="w-full" />
              </div>
              <div>
                <label>γ: {params.gamma.toFixed(2)}</label>
                <input type="range" min={0.1} max={5} step={0.1} value={params.gamma} onChange={updateParam('gamma')} className="w-full" />
              </div>
              <div>
                <label>x0: {params.x0.toFixed(2)}</label>
                <input type="range" min={0.1} max={5} step={0.1} value={params.x0} onChange={updateParam('x0')} className="w-full" />
              </div>
              <div>
                <label>y0: {params.y0.toFixed(2)}</label>
                <input type="range" min={0.1} max={5} step={0.1} value={params.y0} onChange={updateParam('y0')} className="w-full" />
              </div>
            </>
          ) : (
            <>
              <div>
                <label>r: {params.r.toFixed(2)}</label>
                <input type="range" min={0.1} max={4} step={0.1} value={params.r} onChange={updateParam('r')} className="w-full" />
              </div>
              <div>
                <label>K: {params.K.toFixed(2)}</label>
                <input type="range" min={0.5} max={2} step={0.1} value={params.K} onChange={updateParam('K')} className="w-full" />
              </div>
              <div>
                <label>x0: {params.x0.toFixed(2)}</label>
                <input type="range" min={0.01} max={2} step={0.01} value={params.x0} onChange={updateParam('x0')} className="w-full" />
              </div>
            </>
          )}
        </div>
        <div className="flex gap-4">
          <button onClick={reset} className="px-4 py-2 bg-blue-500 text-white rounded">Reset</button>
          <button onClick={togglePlay} className={`px-4 py-2 ${playing ? 'bg-red-500' : 'bg-green-500'} text-white rounded`}>
            {playing ? 'Pause' : 'Play'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default NonlinearEquationsPage;