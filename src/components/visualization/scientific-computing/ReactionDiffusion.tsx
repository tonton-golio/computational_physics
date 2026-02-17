'use client';

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id?: string;
}

const PRESETS: Record<string, { Dp: number; Dq: number; C: number; K: number; label: string }> = {
  standard: { Dp: 1, Dq: 8, C: 4.5, K: 9, label: "Standard" },
  mushrooms: { Dp: 1, Dq: 9, C: 2.3, K: 11, label: "Mushrooms" },
  stripes: { Dp: 1, Dq: 6, C: 3.5, K: 8, label: "Stripes" },
};

export function ReactionDiffusion({}: SimulationProps) {
  const [Dp, setDp] = useState(1);
  const [Dq, setDq] = useState(8);
  const [C, setC] = useState(4.5);
  const [K, setK] = useState(9);
  const [resolution, setResolution] = useState(64);
  const [dt, setDt] = useState(0.01);
  const [isRunning, setIsRunning] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [stepsPerFrame, setStepsPerFrame] = useState(10);
  const [stable, setStable] = useState(true);

  const pRef = useRef<Float64Array | null>(null);
  const qRef = useRef<Float64Array | null>(null);
  const animFrameRef = useRef<number>(0);
  const plotContainerRef = useRef<HTMLDivElement | null>(null);

  // Initialize grids
  const initializeGrids = useCallback(() => {
    const N = resolution;
    const p = new Float64Array(N * N);
    const q = new Float64Array(N * N);

    // Set initial conditions: middle region perturbed
    const low = Math.floor(N * 0.25);
    const high = Math.floor(N * 0.75);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i >= low && i < high && j >= low && j < high) {
          p[i * N + j] = C + 0.1 * (Math.random() - 0.5);
          q[i * N + j] = K / C + 0.2 * (Math.random() - 0.5);
        } else {
          p[i * N + j] = 0;
          q[i * N + j] = 0;
        }
      }
    }

    pRef.current = p;
    qRef.current = q;
    setStepCount(0);
    setStable(true);
  }, [resolution, C, K]);

  useEffect(() => {
    initializeGrids();
  }, [initializeGrids]);

  // Laplacian using rolling (periodic boundary)
  const laplacian = useCallback((X: Float64Array, N: number, dx: number) => {
    const result = new Float64Array(N * N);
    const dx2 = dx * dx;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const idx = i * N + j;
        const ip = ((i + 1) % N) * N + j;
        const im = ((i - 1 + N) % N) * N + j;
        const jp = i * N + (j + 1) % N;
        const jm = i * N + (j - 1 + N) % N;
        result[idx] = (X[ip] + X[im] + X[jp] + X[jm] - 4 * X[idx]) / dx2;
      }
    }
    return result;
  }, []);

  // Forward Euler step
  const evolveStep = useCallback(() => {
    const p = pRef.current;
    const q = qRef.current;
    if (!p || !q) return false;

    const N = resolution;
    const dx = 40 / N;

    const nablaP = laplacian(p, N, dx);
    const nablaQ = laplacian(q, N, dx);

    const pNew = new Float64Array(N * N);
    const qNew = new Float64Array(N * N);

    for (let idx = 0; idx < N * N; idx++) {
      const pVal = p[idx];
      const qVal = q[idx];
      pNew[idx] = pVal + dt * (Dp * nablaP[idx] + pVal * pVal * qVal + C - (K + 1) * pVal);
      qNew[idx] = qVal + dt * (Dq * nablaQ[idx] - pVal * pVal * qVal + K * pVal);

      if (isNaN(pNew[idx]) || isNaN(qNew[idx]) || !isFinite(pNew[idx]) || !isFinite(qNew[idx])) {
        return false; // unstable
      }
    }

    pRef.current = pNew;
    qRef.current = qNew;
    return true;
  }, [resolution, dt, Dp, Dq, C, K, laplacian]);

  // Get 2D array for plotting
  const getGrid = useCallback((arr: Float64Array | null, N: number) => {
    if (!arr) return [];
    const grid: number[][] = [];
    for (let i = 0; i < N; i++) {
      const row: number[] = [];
      for (let j = 0; j < N; j++) {
        row.push(arr[i * N + j]);
      }
      grid.push(row);
    }
    return grid;
  }, []);

  // Animation loop
  const animate = useCallback(() => {
    for (let s = 0; s < stepsPerFrame; s++) {
      const ok = evolveStep();
      if (!ok) {
        setStable(false);
        setIsRunning(false);
        return;
      }
    }
    setStepCount(prev => prev + stepsPerFrame);
    animFrameRef.current = requestAnimationFrame(animate);
  }, [evolveStep, stepsPerFrame]);

  useEffect(() => {
    if (isRunning) {
      animFrameRef.current = requestAnimationFrame(animate);
    }
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isRunning, animate]);

  const handleReset = useCallback(() => {
    setIsRunning(false);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    initializeGrids();
  }, [initializeGrids]);

  const handlePreset = useCallback((key: string) => {
    const p = PRESETS[key];
    setDp(p.Dp);
    setDq(p.Dq);
    setC(p.C);
    setK(p.K);
  }, []);

  const pGrid = useMemo(() => getGrid(pRef.current, resolution), [getGrid, resolution, stepCount]);
  const qGrid = useMemo(() => getGrid(qRef.current, resolution), [getGrid, resolution, stepCount]);

  const heatmapLayout = useMemo(
    () => ({
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(15,15,25,1)',
      font: { color: '#9ca3af', family: 'system-ui', size: 11 },
      margin: { t: 35, r: 10, b: 30, l: 30 },
      xaxis: { showticklabels: false, ticks: '' as const },
      yaxis: { showticklabels: false, ticks: '' as const },
    }),
    []
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <Plot
            data={[
              {
                z: pGrid,
                type: 'heatmap' as const,
                colorscale: 'Hot',
                showscale: true,
                colorbar: { title: { text: 'p', side: 'right' as const }, len: 0.8 },
              },
            ]}
            layout={{
              ...heatmapLayout,
              title: { text: 'Concentration p', font: { size: 14 } },
            } as Partial<Plotly.Layout>}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%', height: '350px' }}
          />
        </div>
        <div>
          <Plot
            data={[
              {
                z: qGrid,
                type: 'heatmap' as const,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: { title: { text: 'q', side: 'right' as const }, len: 0.8 },
              },
            ]}
            layout={{
              ...heatmapLayout,
              title: { text: 'Concentration q', font: { size: 14 } },
            } as Partial<Plotly.Layout>}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%', height: '350px' }}
          />
        </div>
      </div>

      {!stable && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-300 text-sm">
          Simulation became unstable (NaN detected). Try reducing dt or adjusting parameters. Press Reset to restart.
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        <button
          onClick={() => setIsRunning(!isRunning)}
          disabled={!stable}
          className={`px-4 py-2 rounded text-sm text-white disabled:opacity-50 ${isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
        >
          {isRunning ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-600 rounded text-sm hover:bg-gray-700 text-white"
        >
          Reset
        </button>
        {Object.entries(PRESETS).map(([key, val]) => (
          <button
            key={key}
            onClick={() => handlePreset(key)}
            className="px-3 py-2 bg-[#1e1e3a] rounded text-sm hover:bg-[#2a2a4a] text-gray-300"
          >
            {val.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <label className="text-sm text-gray-400">
          D_p: <span className="text-white">{Dp}</span>
          <input
            type="range"
            min={0.1}
            max={10}
            step={0.1}
            value={Dp}
            onChange={(e) => setDp(parseFloat(e.target.value))}
            className="w-full accent-blue-500"
          />
        </label>
        <label className="text-sm text-gray-400">
          D_q: <span className="text-white">{Dq}</span>
          <input
            type="range"
            min={0.1}
            max={15}
            step={0.1}
            value={Dq}
            onChange={(e) => setDq(parseFloat(e.target.value))}
            className="w-full accent-blue-500"
          />
        </label>
        <label className="text-sm text-gray-400">
          C: <span className="text-white">{C.toFixed(1)}</span>
          <input
            type="range"
            min={0.1}
            max={10}
            step={0.1}
            value={C}
            onChange={(e) => setC(parseFloat(e.target.value))}
            className="w-full accent-blue-500"
          />
        </label>
        <label className="text-sm text-gray-400">
          K: <span className="text-white">{K}</span>
          <input
            type="range"
            min={1}
            max={15}
            step={0.5}
            value={K}
            onChange={(e) => setK(parseFloat(e.target.value))}
            className="w-full accent-blue-500"
          />
        </label>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <label className="text-sm text-gray-400">
          dt: <span className="text-white">{dt}</span>
          <select
            value={dt}
            onChange={(e) => setDt(parseFloat(e.target.value))}
            className="w-full bg-[#151525] text-white rounded px-2 py-1 mt-1"
          >
            <option value={0.001}>0.001</option>
            <option value={0.005}>0.005</option>
            <option value={0.01}>0.01</option>
            <option value={0.02}>0.02</option>
            <option value={0.03}>0.03</option>
          </select>
        </label>
        <label className="text-sm text-gray-400">
          Resolution: <span className="text-white">{resolution}</span>
          <select
            value={resolution}
            onChange={(e) => { setResolution(parseInt(e.target.value)); }}
            className="w-full bg-[#151525] text-white rounded px-2 py-1 mt-1"
          >
            <option value={32}>32</option>
            <option value={48}>48</option>
            <option value={64}>64</option>
            <option value={96}>96</option>
          </select>
        </label>
        <label className="text-sm text-gray-400">
          Steps/frame: <span className="text-white">{stepsPerFrame}</span>
          <select
            value={stepsPerFrame}
            onChange={(e) => setStepsPerFrame(parseInt(e.target.value))}
            className="w-full bg-[#151525] text-white rounded px-2 py-1 mt-1"
          >
            <option value={1}>1</option>
            <option value={5}>5</option>
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
          </select>
        </label>
      </div>

      <div className="flex gap-6 text-sm text-gray-400">
        <span>Steps: <span className="text-white">{stepCount}</span></span>
        <span>Time: <span className="text-white">{(stepCount * dt).toFixed(3)}</span></span>
        <span>Status: <span className={stable ? 'text-green-400' : 'text-red-400'}>{stable ? 'Stable' : 'Unstable'}</span></span>
      </div>

      <p className="text-xs text-gray-500">
        Gray-Scott reaction-diffusion model solved with forward Euler. Patterns emerge from the interplay of diffusion rates and reaction kinetics.
        Try different presets to see stripes, spots, and other Turing patterns.
      </p>
    </div>
  );
}
