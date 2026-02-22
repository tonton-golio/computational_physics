"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { clamp } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationMain } from '@/components/ui/simulation-main';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationAux, SimulationLabel, SimulationButton, SimulationCheckbox } from '@/components/ui/simulation-panel';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { linspace } from './ml-utils';

type Optimizer = 'gd' | 'momentum' | 'rmsprop' | 'adam';

interface OptimizerState {
  x: number;
  y: number;
  vx: number;
  vy: number;
  sx: number; // RMSProp / Adam second moment
  sy: number;
  mx: number; // Adam first moment
  my: number;
  t: number;  // Adam timestep
}

const GRID = 80;
const RANGE = 3.5;

/** Loss surface with a global minimum, two local minima, a saddle point, and a plateau. */
function loss(x: number, y: number): number {
  // Global minimum near (1.5, -1.0)
  const global = 0.3 * ((x - 1.5) ** 2 + (y + 1.0) ** 2);
  // Local minimum near (-1.5, 1.5)
  const local1 = 2.0 * Math.exp(-0.8 * ((x + 1.5) ** 2 + (y - 1.5) ** 2));
  // Local minimum near (-1.0, -1.5)
  const local2 = 1.5 * Math.exp(-0.6 * ((x + 1.0) ** 2 + (y + 1.5) ** 2));
  // Saddle near origin
  const saddle = 0.15 * (x * x - y * y);
  // Plateau contribution
  const plateau = 0.5 / (1 + 0.5 * ((x - 2.5) ** 2 + (y - 2.5) ** 2));
  // Base bowl
  const bowl = 0.08 * (x * x + y * y);

  return bowl + saddle + plateau - local1 - local2 + global * 0.4 + 2.0;
}

function gradient(x: number, y: number): [number, number] {
  const eps = 1e-4;
  const gx = (loss(x + eps, y) - loss(x - eps, y)) / (2 * eps);
  const gy = (loss(x, y + eps) - loss(x, y - eps)) / (2 * eps);
  return [gx, gy];
}

function stepOptimizer(
  state: OptimizerState,
  lr: number,
  momentumCoeff: number,
  optimizer: Optimizer,
): OptimizerState {
  const [gx, gy] = gradient(state.x, state.y);
  const s = { ...state };
  const eps = 1e-8;

  switch (optimizer) {
    case 'gd': {
      s.x -= lr * gx;
      s.y -= lr * gy;
      break;
    }
    case 'momentum': {
      s.vx = momentumCoeff * s.vx - lr * gx;
      s.vy = momentumCoeff * s.vy - lr * gy;
      s.x += s.vx;
      s.y += s.vy;
      break;
    }
    case 'rmsprop': {
      const decay = 0.9;
      s.sx = decay * s.sx + (1 - decay) * gx * gx;
      s.sy = decay * s.sy + (1 - decay) * gy * gy;
      s.x -= lr * gx / (Math.sqrt(s.sx) + eps);
      s.y -= lr * gy / (Math.sqrt(s.sy) + eps);
      break;
    }
    case 'adam': {
      const b1 = 0.9, b2 = 0.999;
      s.t += 1;
      s.mx = b1 * s.mx + (1 - b1) * gx;
      s.my = b1 * s.my + (1 - b1) * gy;
      s.sx = b2 * s.sx + (1 - b2) * gx * gx;
      s.sy = b2 * s.sy + (1 - b2) * gy * gy;
      const mxHat = s.mx / (1 - b1 ** s.t);
      const myHat = s.my / (1 - b1 ** s.t);
      const sxHat = s.sx / (1 - b2 ** s.t);
      const syHat = s.sy / (1 - b2 ** s.t);
      s.x -= lr * mxHat / (Math.sqrt(sxHat) + eps);
      s.y -= lr * myHat / (Math.sqrt(syHat) + eps);
      break;
    }
  }

  // Clamp to grid
  s.x = clamp(s.x, -RANGE, RANGE);
  s.y = clamp(s.y, -RANGE, RANGE);
  return s;
}

function makeInitState(x: number, y: number): OptimizerState {
  return { x, y, vx: 0, vy: 0, sx: 0, sy: 0, mx: 0, my: 0, t: 0 };
}

/** Canvas overlay that draws the optimizer trail on top of the heatmap. */
function TrailOverlay({
  trail,
  gridRange,
  gridSize,
}: {
  trail: [number, number][];
  gridRange: number;
  gridSize: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || trail.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match parent dimensions
    const parent = canvas.parentElement;
    if (!parent) return;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    ctx.clearRect(0, 0, w, h);

    // Map from data coordinates to canvas pixels
    // Heatmap has margins: t=20, r=20, b=45, l=55
    const ml = 55, mr = 20, mt = 20, mb = 45;
    const plotW = w - ml - mr;
    const plotH = h - mt - mb;
    const toX = (v: number) => ml + ((v + gridRange) / (2 * gridRange)) * plotW;
    const toY = (v: number) => mt + plotH - ((v + gridRange) / (2 * gridRange)) * plotH;

    // Draw trail line
    if (trail.length > 1) {
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toX(trail[0][0]), toY(trail[0][1]));
      for (let i = 1; i < trail.length; i++) {
        ctx.lineTo(toX(trail[i][0]), toY(trail[i][1]));
      }
      ctx.stroke();
    }

    // Start marker
    ctx.fillStyle = '#22d3ee';
    ctx.beginPath();
    ctx.arc(toX(trail[0][0]), toY(trail[0][1]), 6, 0, Math.PI * 2);
    ctx.fill();

    // Current position
    const last = trail[trail.length - 1];
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(toX(last[0]), toY(last[1]), 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [trail, gridRange, gridSize]);

  return (
    <canvas
      ref={ref}
      className="pointer-events-none absolute inset-0"
      style={{ zIndex: 10 }}
    />
  );
}

export default function LossLandscapeExplorer({}: SimulationComponentProps): React.ReactElement {
  const [lr, setLr] = useState(0.05);
  const [momentumCoeff, setMomentumCoeff] = useState(0.9);
  const [optimizer, setOptimizer] = useState<Optimizer>('adam');
  const [trail, setTrail] = useState<[number, number][]>([]);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [running, setRunning] = useState(false);
  const [showContour, setShowContour] = useState(false);
  const stateRef = useRef<OptimizerState>(makeInitState(-2.5, 2.5));
  const rafRef = useRef<number>(0);
  const stepCountRef = useRef(0);

  // Compute landscape grid
  const grid = useMemo(() => {
    const xs = linspace(-RANGE, RANGE, GRID);
    const ys = linspace(-RANGE, RANGE, GRID);
    const z = ys.map((yy) => xs.map((xx) => loss(xx, yy)));
    return { x: xs, y: ys, z };
  }, []);

  const dropBall = useCallback(() => {
    const rx = (Math.random() - 0.5) * 2 * RANGE;
    const ry = (Math.random() - 0.5) * 2 * RANGE;
    stateRef.current = makeInitState(rx, ry);
    setTrail([[rx, ry]]);
    setLossHistory([loss(rx, ry)]);
    setRunning(false);
    stepCountRef.current = 0;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  const doStep = useCallback(() => {
    stateRef.current = stepOptimizer(stateRef.current, lr, momentumCoeff, optimizer);
    stepCountRef.current += 1;
    const { x, y } = stateRef.current;
    setTrail((prev) => [...prev, [x, y]]);
    setLossHistory((prev) => [...prev, loss(x, y)]);
  }, [lr, momentumCoeff, optimizer]);

  const runSteps = useCallback((n: number) => {
    setRunning(true);
    let count = 0;
    const tick = () => {
      if (count >= n) {
        setRunning(false);
        return;
      }
      doStep();
      count++;
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [doStep]);

  const failureDiverge = useCallback(() => {
    stateRef.current = makeInitState(0.1, 0.0);
    setTrail([[0.1, 0.0]]);
    setLossHistory([loss(0.1, 0.0)]);
    setRunning(false);
    stepCountRef.current = 0;
    setLr(1.5);
  }, []);

  const failureStuck = useCallback(() => {
    stateRef.current = makeInitState(-1.5, 1.5);
    setTrail([[-1.5, 1.5]]);
    setLossHistory([loss(-1.5, 1.5)]);
    setRunning(false);
    stepCountRef.current = 0;
    setLr(0.001);
  }, []);

  useEffect(() => {
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  return (
    <SimulationPanel title="Loss Landscape Explorer">
      <SimulationSettings>
        <div className="flex flex-wrap gap-2">
          <SimulationButton variant="primary" onClick={dropBall}>
            Drop Ball (random)
          </SimulationButton>
          <SimulationButton variant="primary" onClick={() => doStep()} disabled={running}>
            Step
          </SimulationButton>
          <SimulationButton variant="primary" onClick={() => runSteps(200)} disabled={running}>
            Run 200 Steps
          </SimulationButton>
          <SimulationButton variant="danger" onClick={failureDiverge}>
            Divergence Demo
          </SimulationButton>
          <SimulationButton variant="secondary" onClick={failureStuck}>
            Getting Stuck Demo
          </SimulationButton>
        </div>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-2">
          <div>
            <SimulationLabel>Optimizer</SimulationLabel>
            <select
              value={optimizer}
              onChange={(e) => setOptimizer(e.target.value as Optimizer)}
              className="mt-1 w-full rounded bg-[var(--surface-2,#27272a)] px-3 py-1.5 text-sm text-[var(--text-strong)]"
            >
              <option value="gd">Gradient Descent</option>
              <option value="momentum">Momentum</option>
              <option value="rmsprop">RMSProp</option>
              <option value="adam">Adam</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <SimulationCheckbox checked={showContour} onChange={setShowContour} label="Show loss curve" />
          </div>
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <SimulationLabel>
              Learning rate: {lr.toFixed(4)}
            </SimulationLabel>
            <Slider
              min={0.001}
              max={2.0}
              step={0.001}
              value={[lr]}
              onValueChange={([v]) => setLr(v)}
            />
          </div>
          <div>
            <SimulationLabel>
              Momentum: {momentumCoeff.toFixed(2)}
            </SimulationLabel>
            <Slider
              min={0}
              max={0.99}
              step={0.01}
              value={[momentumCoeff]}
              onValueChange={([v]) => setMomentumCoeff(v)}
            />
          </div>
        </div>
      </SimulationConfig>

      <div className={showContour ? 'grid grid-cols-1 gap-4 md:grid-cols-2' : ''}>
        <SimulationMain scaleMode="contain" className="relative">
          {/* Heatmap landscape */}
          <CanvasHeatmap
            data={[{
              z: grid.z,
              x: grid.x,
              y: grid.y,
              colorscale: 'Viridis',
              showscale: false,
            }]}
            layout={{
              xaxis: { title: { text: 'w\u2081' } },
              yaxis: { title: { text: 'w\u2082' } },
              margin: { t: 20, r: 20, b: 45, l: 55 },
            }}
            style={{ width: '100%', height: 420 }}
          />
          {/* Optimizer trail overlay */}
          {trail.length > 0 && (
            <TrailOverlay trail={trail} gridRange={RANGE} gridSize={GRID} />
          )}
        </SimulationMain>

        {showContour && lossHistory.length > 0 && (
          <SimulationAux>
            <CanvasChart
              data={[
                {
                  x: Array.from({ length: lossHistory.length }, (_, i) => i),
                  y: lossHistory,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Loss',
                  line: { color: '#ef4444', width: 2 },
                },
              ]}
              layout={{
                xaxis: { title: { text: 'Step' } },
                yaxis: { title: { text: 'Loss' } },
                margin: { t: 20, r: 20, b: 45, l: 55 },
              }}
              style={{ width: '100%', height: 420 }}
            />
          </SimulationAux>
        )}
      </div>

      <SimulationResults>
        <span className="text-xs text-[var(--text-muted)]">
          Steps: {stepCountRef.current}
        </span>
        {trail.length > 0 && !running && stepCountRef.current >= 50 && (
          <div className="mt-3 rounded bg-[var(--surface-2,#27272a)] p-3 text-sm text-[var(--text-muted)]">
            Final loss: {lossHistory[lossHistory.length - 1]?.toFixed(4)} after{' '}
            {stepCountRef.current} steps with {optimizer.toUpperCase()}.
            {optimizer === 'adam' && lossHistory[lossHistory.length - 1] < 1.2 &&
              ' Adam adapts per-parameter learning rates, making it the default optimizer in modern deep learning.'}
          </div>
        )}
      </SimulationResults>
    </SimulationPanel>
  );
}
