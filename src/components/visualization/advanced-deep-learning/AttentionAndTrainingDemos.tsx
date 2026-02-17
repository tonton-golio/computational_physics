'use client';

import { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function linspace(start: number, end: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => start + ((end - start) * i) / (n - 1));
}

export function AttentionHeatmapDemo() {
  const [tokens, setTokens] = useState(10);
  const [focus, setFocus] = useState(0.5);

  const z = useMemo(() => {
    const c = focus * (tokens - 1);
    return Array.from({ length: tokens }, (_, i) =>
      Array.from({ length: tokens }, (_, j) => {
        const dist = Math.abs(j - c) / Math.max(tokens - 1, 1);
        const local = Math.exp(-6 * dist * dist);
        const causal = j <= i ? 1 : 0.08;
        return local * causal;
      })
    );
  }, [tokens, focus]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8 space-y-4">
      <h3 className="text-xl font-semibold text-white">Transformer Attention Heatmap</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-gray-300">Token count: {tokens}</label>
          <input className="w-full" type="range" min={6} max={20} step={1} value={tokens} onChange={(e) => setTokens(parseInt(e.target.value, 10))} />
        </div>
        <div>
          <label className="text-sm text-gray-300">Focus position: {focus.toFixed(2)}</label>
          <input className="w-full" type="range" min={0} max={1} step={0.01} value={focus} onChange={(e) => setFocus(parseFloat(e.target.value))} />
        </div>
      </div>
      <Plot
        data={[{ z, type: 'heatmap', colorscale: 'Viridis' }]}
        layout={{
          title: { text: 'Causal self-attention weights' },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          margin: { t: 40, r: 20, b: 40, l: 50 },
          xaxis: { title: { text: 'Key token' }, gridcolor: '#1e1e2e' },
          yaxis: { title: { text: 'Query token' }, gridcolor: '#1e1e2e' },
          height: 420,
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: 420 }}
      />
    </div>
  );
}

export function OptimizerTrajectoryDemo() {
  const [lr, setLr] = useState(0.08);
  const [momentum, setMomentum] = useState(0.7);

  const { xs, ys, loss } = useMemo(() => {
    const grad = (x: number, y: number) => [0.8 * x + 0.25 * y, 0.25 * x + 1.2 * y] as const;
    let x = 2.6;
    let y = -2.2;
    let vx = 0;
    let vy = 0;
    const xs: number[] = [x];
    const ys: number[] = [y];
    const loss: number[] = [0.4 * x * x + 0.25 * x * y + 0.6 * y * y];
    for (let i = 0; i < 40; i++) {
      const [gx, gy] = grad(x, y);
      vx = momentum * vx + gx;
      vy = momentum * vy + gy;
      x -= lr * vx;
      y -= lr * vy;
      xs.push(x);
      ys.push(y);
      loss.push(0.4 * x * x + 0.25 * x * y + 0.6 * y * y);
    }
    return { xs, ys, loss };
  }, [lr, momentum]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8 space-y-4">
      <h3 className="text-xl font-semibold text-white">Optimizer Trajectory on Loss Basin</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-gray-300">Learning rate: {lr.toFixed(3)}</label>
          <input className="w-full" type="range" min={0.01} max={0.2} step={0.005} value={lr} onChange={(e) => setLr(parseFloat(e.target.value))} />
        </div>
        <div>
          <label className="text-sm text-gray-300">Momentum: {momentum.toFixed(2)}</label>
          <input className="w-full" type="range" min={0} max={0.95} step={0.01} value={momentum} onChange={(e) => setMomentum(parseFloat(e.target.value))} />
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Plot
          data={[
            {
              x: xs,
              y: ys,
              mode: 'lines+markers',
              type: 'scatter',
              marker: { size: 5, color: xs.map((_, i) => i), colorscale: 'Plasma' },
              line: { color: '#3b82f6', width: 2 },
              name: 'Trajectory',
            },
          ]}
          layout={{
            title: { text: 'Parameter trajectory' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            xaxis: { title: { text: 'w1' }, gridcolor: '#1e1e2e' },
            yaxis: { title: { text: 'w2' }, gridcolor: '#1e1e2e' },
            height: 360,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 360 }}
        />
        <Plot
          data={[{ x: linspace(0, loss.length - 1, loss.length), y: loss, mode: 'lines+markers', type: 'scatter', line: { color: '#10b981', width: 2 } }]}
          layout={{
            title: { text: 'Loss by iteration' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            xaxis: { title: { text: 'Iteration' }, gridcolor: '#1e1e2e' },
            yaxis: { title: { text: 'Loss' }, gridcolor: '#1e1e2e' },
            height: 360,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 360 }}
        />
      </div>
    </div>
  );
}

export function LatentInterpolationDemo() {
  const [alpha, setAlpha] = useState(0.5);

  const curve = useMemo(() => {
    const t = linspace(0, 1, 120);
    const start = [-2, 0.4];
    const end = [2, -0.3];
    const x = t.map((v) => (1 - v) * start[0] + v * end[0]);
    const y = t.map((v) => (1 - v) * start[1] + v * end[1] + 0.2 * Math.sin(8 * v));
    const px = (1 - alpha) * start[0] + alpha * end[0];
    const py = (1 - alpha) * start[1] + alpha * end[1] + 0.2 * Math.sin(8 * alpha);
    return { x, y, px, py };
  }, [alpha]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8 space-y-4">
      <h3 className="text-xl font-semibold text-white">Latent Space Interpolation (VAE Intuition)</h3>
      <div>
        <label className="text-sm text-gray-300">Interpolation factor α: {alpha.toFixed(2)}</label>
        <input className="w-full" type="range" min={0} max={1} step={0.01} value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))} />
      </div>
      <Plot
        data={[
          { x: curve.x, y: curve.y, type: 'scatter', mode: 'lines', line: { color: '#8b5cf6', width: 2 }, name: 'Latent path' },
          { x: [curve.px], y: [curve.py], type: 'scatter', mode: 'markers', marker: { size: 10, color: '#ec4899' }, name: 'Decoded sample' },
        ]}
        layout={{
          title: { text: 'Smooth semantic interpolation through latent manifold' },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          xaxis: { title: { text: 'z1' }, gridcolor: '#1e1e2e' },
          yaxis: { title: { text: 'z2' }, gridcolor: '#1e1e2e' },
          height: 380,
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: 380 }}
      />
    </div>
  );
}
