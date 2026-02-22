"use client";

import { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationConfig, SimulationSettings, SimulationLabel, SimulationCheckbox } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';

function linspace(start: number, end: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => start + ((end - start) * i) / (n - 1));
}

// ---- Attention Heatmap Demo ----

const EXAMPLE_TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'warm', 'mat', 'and', 'purred', 'softly',
  'while', 'the', 'rain', 'fell', 'gently', 'outside', 'the', 'old', 'window', 'pane'];

export function AttentionHeatmapDemo() {
  const [tokens, setTokens] = useState(10);
  const [focus, setFocus] = useState(0.5);
  const [numHeads, setNumHeads] = useState(1);
  const tokenLabels = EXAMPLE_TOKENS.slice(0, tokens);

  const heads = useMemo(() => {
    const result: number[][][] = [];
    for (let h = 0; h < numHeads; h++) {
      const c = focus * (tokens - 1);
      // Each head has a different attention pattern
      const headOffset = h * 2.5;
      const spreadFactor = 4 + h * 3;
      const z = Array.from({ length: tokens }, (_, i) =>
        Array.from({ length: tokens }, (_, j) => {
          const dist = Math.abs(j - c + headOffset) / Math.max(tokens - 1, 1);
          const local = Math.exp(-spreadFactor * dist * dist);
          const causal = j <= i ? 1 : 0.08;
          return local * causal;
        })
      );
      result.push(z);
    }
    return result;
  }, [tokens, focus, numHeads]);

  const headLabels = ['Head 1 (local)', 'Head 2 (broad)', 'Head 3 (distant)'];

  return (
    <SimulationPanel title="Transformer Attention Heatmap">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
          <div>
            <SimulationLabel className="text-sm text-[var(--text-muted)]">Token count: {tokens}</SimulationLabel>
            <Slider min={6} max={20} step={1} value={[tokens]} onValueChange={([v]) => setTokens(v)} />
          </div>
          <div>
            <SimulationLabel className="text-sm text-[var(--text-muted)]">Focus position: {focus.toFixed(2)}</SimulationLabel>
            <Slider min={0} max={1} step={0.01} value={[focus]} onValueChange={([v]) => setFocus(v)} />
          </div>
          <div>
            <SimulationLabel className="text-sm text-[var(--text-muted)]">Attention heads: {numHeads}</SimulationLabel>
            <Slider min={1} max={3} step={1} value={[numHeads]} onValueChange={([v]) => setNumHeads(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className={`grid gap-4 ${numHeads === 1 ? 'grid-cols-1' : numHeads === 2 ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1 lg:grid-cols-3'}`}>
          {heads.map((z, h) => (
            <div key={h}>
              {numHeads > 1 && (
                <p className="text-center text-sm text-[var(--text-muted)] mb-1 font-semibold">
                  {headLabels[h]}
                </p>
              )}
              <CanvasHeatmap
                data={[{
                  z,
                  type: 'heatmap',
                  colorscale: 'Viridis',
                  x: tokenLabels,
                  y: tokenLabels,
                  showscale: h === 0,
                }]}
                layout={{
                  title: numHeads === 1 ? { text: 'Causal self-attention weights' } : undefined,
                  margin: { t: numHeads === 1 ? 40 : 20, r: 20, b: 60, l: 70 },
                  xaxis: { title: { text: 'Key token' } },
                  yaxis: { title: { text: 'Query token' } },
                }}
                style={{ width: '100%', height: numHeads === 1 ? 420 : 350 }}
              />
            </div>
          ))}
        </div>
      </SimulationMain>
      <div className="p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        The causal mask ensures each token can only attend to itself and previous tokens (lower-left triangle).
        {numHeads > 1 && ' Different heads learn different attention patterns: Head 1 focuses locally, Head 2 has broader context, and Head 3 attends to more distant tokens.'}
      </div>
    </SimulationPanel>
  );
}

// ---- Optimizer Trajectory Demo ----

type OptimizerType = 'sgd' | 'sgd-momentum' | 'adam';

function runOptimizer(type: OptimizerType, lr: number, momentum: number) {
  const lossFunc = (x: number, y: number) => 0.4 * x * x + 0.25 * x * y + 0.6 * y * y;
  const grad = (x: number, y: number) => [0.8 * x + 0.25 * y, 0.25 * x + 1.2 * y] as const;

  let x = 2.6, y = -2.2;
  let vx = 0, vy = 0;
  let mx = 0, my = 0, vvx = 0, vvy = 0;
  const xs: number[] = [x], ys: number[] = [y];
  const loss: number[] = [lossFunc(x, y)];

  for (let i = 0; i < 40; i++) {
    const [gx, gy] = grad(x, y);
    if (type === 'sgd') {
      x -= lr * gx;
      y -= lr * gy;
    } else if (type === 'sgd-momentum') {
      vx = momentum * vx + gx;
      vy = momentum * vy + gy;
      x -= lr * vx;
      y -= lr * vy;
    } else {
      // Adam
      const t = i + 1;
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      mx = b1 * mx + (1 - b1) * gx;
      my = b1 * my + (1 - b1) * gy;
      vvx = b2 * vvx + (1 - b2) * gx * gx;
      vvy = b2 * vvy + (1 - b2) * gy * gy;
      const mxh = mx / (1 - Math.pow(b1, t));
      const myh = my / (1 - Math.pow(b1, t));
      const vxh = vvx / (1 - Math.pow(b2, t));
      const vyh = vvy / (1 - Math.pow(b2, t));
      x -= lr * mxh / (Math.sqrt(vxh) + eps);
      y -= lr * myh / (Math.sqrt(vyh) + eps);
    }
    xs.push(x);
    ys.push(y);
    loss.push(lossFunc(x, y));
  }
  return { xs, ys, loss };
}

const OPTIMIZER_COLORS: Record<OptimizerType, string> = {
  'sgd': '#ef4444',
  'sgd-momentum': '#3b82f6',
  'adam': '#10b981',
};

export function OptimizerTrajectoryDemo() {
  const [lr, setLr] = useState(0.08);
  const [momentum, setMomentum] = useState(0.7);
  const [selectedOptimizers, setSelectedOptimizers] = useState<Set<OptimizerType>>(
    new Set(['sgd-momentum'])
  );
  const [showContours, setShowContours] = useState(true);
  const trajectories = useMemo(() => {
    const result: Record<OptimizerType, ReturnType<typeof runOptimizer>> = {} as any;
    for (const opt of ['sgd', 'sgd-momentum', 'adam'] as OptimizerType[]) {
      if (selectedOptimizers.has(opt)) {
        result[opt] = runOptimizer(opt, lr, momentum);
      }
    }
    return result;
  }, [lr, momentum, selectedOptimizers]);

  // Contour data for loss surface
  const contourData = useMemo(() => {
    if (!showContours) return null;
    const range = linspace(-3, 3, 50);
    const z = range.map(yi => range.map(xi => 0.4 * xi * xi + 0.25 * xi * yi + 0.6 * yi * yi));
    return { x: range, y: range, z };
  }, [showContours]);

  const toggleOptimizer = (opt: OptimizerType) => {
    setSelectedOptimizers(prev => {
      const next = new Set(prev);
      if (next.has(opt)) next.delete(opt);
      else next.add(opt);
      return next;
    });
  };

  const trajectoryTraces: any[] = [];

  // Show loss contour reference circles when enabled
  if (contourData) {
    const levels = [0.5, 1, 2, 3, 5];
    for (const level of levels) {
      const pts = 100;
      const cx: number[] = [];
      const cy: number[] = [];
      for (let i = 0; i <= pts; i++) {
        const theta = (2 * Math.PI * i) / pts;
        const r = Math.sqrt(level / 0.5);
        cx.push(r * Math.cos(theta));
        cy.push(r * Math.sin(theta));
      }
      trajectoryTraces.push({
        x: cx,
        y: cy,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: 'rgba(107, 33, 168, 0.3)', width: 1 },
        showlegend: false,
        hoverinfo: 'skip',
      });
    }
  }

  for (const [opt, traj] of Object.entries(trajectories)) {
    const color = OPTIMIZER_COLORS[opt as OptimizerType];
    const label = opt === 'sgd' ? 'SGD' : opt === 'sgd-momentum' ? 'SGD + Momentum' : 'Adam';
    trajectoryTraces.push({
      x: traj.xs,
      y: traj.ys,
      mode: 'lines+markers',
      type: 'scatter',
      marker: { size: 4, color },
      line: { color, width: 2 },
      name: label,
    });
  }

  const lossTraces: any[] = [];
  for (const [opt, traj] of Object.entries(trajectories)) {
    const color = OPTIMIZER_COLORS[opt as OptimizerType];
    const label = opt === 'sgd' ? 'SGD' : opt === 'sgd-momentum' ? 'SGD + Momentum' : 'Adam';
    lossTraces.push({
      x: linspace(0, traj.loss.length - 1, traj.loss.length),
      y: traj.loss,
      mode: 'lines',
      type: 'scatter',
      line: { color, width: 2 },
      name: label,
    });
  }

  return (
    <SimulationPanel title="Optimizer Trajectory Comparison">
      <SimulationSettings>
        <div className="flex flex-wrap gap-3">
          {(['sgd', 'sgd-momentum', 'adam'] as OptimizerType[]).map(opt => {
            const label = opt === 'sgd' ? 'SGD' : opt === 'sgd-momentum' ? 'SGD + Momentum' : 'Adam';
            return (
              <SimulationCheckbox key={opt} checked={selectedOptimizers.has(opt)} onChange={() => toggleOptimizer(opt)} label={label} style={{ color: selectedOptimizers.has(opt) ? OPTIMIZER_COLORS[opt] : '#666' }} />
            );
          })}
          <SimulationCheckbox checked={showContours} onChange={setShowContours} label="Show contours" />
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
          <div>
            <SimulationLabel className="text-sm text-[var(--text-muted)]">Learning rate: {lr.toFixed(3)}</SimulationLabel>
            <Slider min={0.01} max={0.2} step={0.005} value={[lr]} onValueChange={([v]) => setLr(v)} />
          </div>
          <div>
            <SimulationLabel className="text-sm text-[var(--text-muted)]">Momentum: {momentum.toFixed(2)}</SimulationLabel>
            <Slider min={0} max={0.95} step={0.01} value={[momentum]} onValueChange={([v]) => setMomentum(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasChart
            data={trajectoryTraces}
            layout={{
              title: { text: 'Parameter trajectory' },
              xaxis: { title: { text: 'w1' }, range: [-3, 3] },
              yaxis: { title: { text: 'w2' }, range: [-3, 3] },
            }}
            style={{ width: '100%', height: 380 }}
          />
          <CanvasChart
            data={lossTraces}
            layout={{
              title: { text: 'Loss by iteration' },
              xaxis: { title: { text: 'Iteration' } },
              yaxis: { title: { text: 'Loss' }, type: 'log' },
            }}
            style={{ width: '100%', height: 380 }}
          />
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}

// ---- Latent Interpolation Demo ----


function decode2D(z1: number, z2: number): number {
  // Simulated decoder: maps latent coordinates to a "decoded value"
  return Math.sin(z1 * 1.5) * Math.cos(z2 * 1.5) + 0.3 * Math.sin(2 * z1 + z2);
}

export function LatentInterpolationDemo() {
  const [alpha, setAlpha] = useState(0.5);
  const [showGrid, setShowGrid] = useState(true);
  const gridData = useMemo(() => {
    const gridSize = 20;
    const z1Range = linspace(-3, 3, gridSize);
    const z2Range = linspace(-3, 3, gridSize);
    const values = z2Range.map(z2 => z1Range.map(z1 => decode2D(z1, z2)));
    return { z1Range, z2Range, values };
  }, []);

  const path = useMemo(() => {
    const start = [-2, 0.4];
    const end = [2, -0.3];
    const t = linspace(0, 1, 120);
    const x = t.map(v => (1 - v) * start[0] + v * end[0]);
    const y = t.map(v => (1 - v) * start[1] + v * end[1] + 0.2 * Math.sin(8 * v));
    const px = (1 - alpha) * start[0] + alpha * end[0];
    const py = (1 - alpha) * start[1] + alpha * end[1] + 0.2 * Math.sin(8 * alpha);
    return { x, y, px, py, start, end };
  }, [alpha]);

  const scatterTraces: any[] = [
    {
      x: path.x,
      y: path.y,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#8b5cf6', width: 3 },
      name: 'Interpolation path',
    },
    {
      x: [path.start[0]],
      y: [path.start[1]],
      type: 'scatter',
      mode: 'markers',
      marker: { size: 12, color: '#22c55e', symbol: 'diamond' },
      name: 'Start point',
    },
    {
      x: [path.end[0]],
      y: [path.end[1]],
      type: 'scatter',
      mode: 'markers',
      marker: { size: 12, color: '#ef4444', symbol: 'diamond' },
      name: 'End point',
    },
    {
      x: [path.px],
      y: [path.py],
      type: 'scatter',
      mode: 'markers',
      marker: { size: 14, color: '#ec4899', symbol: 'circle' },
      name: `Sample (alpha=${alpha.toFixed(2)})`,
    },
  ];

  return (
    <SimulationPanel title="Latent Space Interpolation (VAE Intuition)">
      <SimulationSettings>
        <SimulationLabel className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer">
          <input
            type="checkbox"
            checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)}
            className="accent-blue-500"
          />
          Show 2D decoded grid
        </SimulationLabel>
      </SimulationSettings>
      <SimulationConfig>
        <div className="flex-1 min-w-[200px]">
          <SimulationLabel className="text-sm text-[var(--text-muted)]">Interpolation factor alpha: {alpha.toFixed(2)}</SimulationLabel>
          <Slider min={0} max={1} step={0.01} value={[alpha]} onValueChange={([v]) => setAlpha(v)} />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {showGrid && (
            <CanvasHeatmap
              data={[{ z: gridData.values, x: gridData.z1Range, y: gridData.z2Range, colorscale: 'Viridis' }]}
              layout={{ title: { text: 'Decoded Latent Grid' }, xaxis: { title: { text: 'z1' } }, yaxis: { title: { text: 'z2' } } }}
              style={{ width: '100%', height: 450 }}
            />
          )}
          <CanvasChart
            data={scatterTraces}
            layout={{
              title: { text: 'Interpolation through latent manifold' },
              xaxis: { title: { text: 'z1' }, range: [-3.5, 3.5] },
              yaxis: { title: { text: 'z2' }, range: [-3.5, 3.5] },
            }}
            style={{ width: '100%', height: 450 }}
          />
        </div>
      </SimulationMain>
      <div className="p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        The 2D grid shows decoded values across the latent space. A well-trained VAE produces smooth transitions: nearby latent points decode to similar outputs.
        The interpolation path demonstrates that moving linearly through latent space produces semantically meaningful intermediate samples.
      </div>
    </SimulationPanel>
  );
}
