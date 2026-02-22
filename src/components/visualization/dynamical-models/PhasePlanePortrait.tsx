"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Interactive 2D phase portrait for a predator-prey (Lotka-Volterra) system.
 *
 * dx/dt = alpha * x - beta * x * y
 * dy/dt = delta * x * y - gamma * y
 *
 * Shows nullclines, vector field arrows, and multiple trajectories.
 */

function rk4Step(
  x: number, y: number, dt: number,
  alpha: number, beta: number, gamma: number, delta: number,
): [number, number] {
  const fx = (xx: number, yy: number) => alpha * xx - beta * xx * yy;
  const fy = (xx: number, yy: number) => delta * xx * yy - gamma * yy;

  const k1x = fx(x, y);
  const k1y = fy(x, y);
  const k2x = fx(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
  const k2y = fy(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
  const k3x = fx(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
  const k3y = fy(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
  const k4x = fx(x + dt * k3x, y + dt * k3y);
  const k4y = fy(x + dt * k3x, y + dt * k3y);

  return [
    x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x),
    y + (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y),
  ];
}

function integrateTrajectory(
  x0: number, y0: number,
  alpha: number, beta: number, gamma: number, delta: number,
  dt: number, steps: number,
): { xs: number[]; ys: number[] } {
  const xs: number[] = [x0];
  const ys: number[] = [y0];
  let x = x0;
  let y = y0;
  for (let i = 0; i < steps; i++) {
    [x, y] = rk4Step(x, y, dt, alpha, beta, gamma, delta);
    if (!isFinite(x) || !isFinite(y) || x < 0 || y < 0 || x > 100 || y > 100) break;
    xs.push(x);
    ys.push(y);
  }
  return { xs, ys };
}

export default function PhasePlanePortrait({}: SimulationComponentProps) {
  const [alpha, setAlpha] = useState(1.0);
  const [beta, setBeta] = useState(0.1);
  const [gamma, setGamma] = useState(1.5);
  const [delta, setDelta] = useState(0.075);

  // Fixed point
  const fpX = gamma / delta;
  const fpY = alpha / beta;

  // Nullclines
  const nullclineData = useMemo(() => {
    const xMax = Math.max(fpX * 3, 10);
    const yMax = Math.max(fpY * 3, 10);
    const n = 200;

    // x-nullcline: dx/dt = 0 => y = alpha/beta (horizontal line) and x = 0 (y-axis)
    const xNullX: number[] = [];
    const xNullY: number[] = [];
    for (let i = 0; i <= n; i++) {
      xNullX.push((i / n) * xMax);
      xNullY.push(fpY);
    }

    // y-nullcline: dy/dt = 0 => x = gamma/delta (vertical line) and y = 0 (x-axis)
    const yNullX: number[] = [];
    const yNullY: number[] = [];
    for (let i = 0; i <= n; i++) {
      yNullX.push(fpX);
      yNullY.push((i / n) * yMax);
    }

    return { xNullX, xNullY, yNullX, yNullY, xMax, yMax };
  }, [fpX, fpY]);

  // Vector field as arrows (using short line segments)
  const vectorField = useMemo(() => {
    const { xMax, yMax } = nullclineData;
    const nx = 16;
    const ny = 16;
    const arrowX: number[] = [];
    const arrowY: number[] = [];

    for (let i = 1; i <= nx; i++) {
      for (let j = 1; j <= ny; j++) {
        const x = (i / (nx + 1)) * xMax;
        const y = (j / (ny + 1)) * yMax;

        const dx = alpha * x - beta * x * y;
        const dy = delta * x * y - gamma * y;
        const mag = Math.sqrt(dx * dx + dy * dy);
        if (mag < 1e-10) continue;

        const scale = Math.min(xMax / (nx * 3), yMax / (ny * 3));
        const normScale = scale / Math.pow(mag, 0.5);

        const ex = x + dx * normScale;
        const ey = y + dy * normScale;

        arrowX.push(x, ex, NaN);
        arrowY.push(y, ey, NaN);
      }
    }

    return { arrowX, arrowY };
  }, [alpha, beta, gamma, delta, nullclineData]);

  // Trajectories from several initial conditions
  const trajectories = useMemo(() => {
    const { xMax, yMax } = nullclineData;
    const ics = [
      [fpX * 0.5, fpY * 0.5],
      [fpX * 1.5, fpY * 1.5],
      [fpX * 0.3, fpY * 1.8],
      [fpX * 2.0, fpY * 0.3],
    ];
    const colors = ['#3b82f6', '#22c55e', '#f97316', '#a855f7'];
    return ics.map(([x0, y0], idx) => {
      const ic = integrateTrajectory(
        Math.min(x0, xMax * 0.9),
        Math.min(y0, yMax * 0.9),
        alpha, beta, gamma, delta,
        0.02, 3000,
      );
      return {
        xs: ic.xs,
        ys: ic.ys,
        color: colors[idx],
      };
    });
  }, [alpha, beta, gamma, delta, nullclineData, fpX, fpY]);

  const chartData = useMemo(() => {
    const traces: any[] = [];

    // Vector field
    traces.push({
      x: vectorField.arrowX,
      y: vectorField.arrowY,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#6b7280', width: 1 },
      showlegend: false,
    });

    // x-nullcline
    traces.push({
      x: nullclineData.xNullX,
      y: nullclineData.xNullY,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#ef4444', width: 2, dash: 'dash' },
      name: 'x-nullcline (y = \u03B1/\u03B2)',
    });

    // y-nullcline
    traces.push({
      x: nullclineData.yNullX,
      y: nullclineData.yNullY,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#3b82f6', width: 2, dash: 'dash' },
      name: 'y-nullcline (x = \u03B3/\u03B4)',
    });

    // Trajectories
    for (const traj of trajectories) {
      traces.push({
        x: traj.xs,
        y: traj.ys,
        type: 'scatter',
        mode: 'lines',
        line: { color: traj.color, width: 2 },
        showlegend: false,
      });
      // Start marker
      traces.push({
        x: [traj.xs[0]],
        y: [traj.ys[0]],
        type: 'scatter',
        mode: 'markers',
        marker: { color: traj.color, size: 7 },
        showlegend: false,
      });
    }

    // Fixed point
    traces.push({
      x: [fpX],
      y: [fpY],
      type: 'scatter',
      mode: 'markers',
      marker: { color: '#ffffff', size: 9, line: { width: 2, color: '#ef4444' } },
      name: 'Fixed point',
    });

    return traces;
  }, [vectorField, nullclineData, trajectories, fpX, fpY]);

  return (
    <SimulationPanel title="Phase Plane Portrait: Predator-Prey">
      <SimulationConfig>
        <div>
          <SimulationLabel>Prey growth &alpha;: {alpha.toFixed(2)}</SimulationLabel>
          <Slider value={[alpha]} onValueChange={([v]) => setAlpha(v)} min={0.1} max={3} step={0.05} />
        </div>
        <div>
          <SimulationLabel>Predation &beta;: {beta.toFixed(3)}</SimulationLabel>
          <Slider value={[beta]} onValueChange={([v]) => setBeta(v)} min={0.01} max={0.5} step={0.005} />
        </div>
        <div>
          <SimulationLabel>Predator death &gamma;: {gamma.toFixed(2)}</SimulationLabel>
          <Slider value={[gamma]} onValueChange={([v]) => setGamma(v)} min={0.1} max={3} step={0.05} />
        </div>
        <div>
          <SimulationLabel>Pred. efficiency &delta;: {delta.toFixed(3)}</SimulationLabel>
          <Slider value={[delta]} onValueChange={([v]) => setDelta(v)} min={0.01} max={0.3} step={0.005} />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <CanvasChart
        data={chartData}
        layout={{
          height: 480,
          margin: { t: 20, b: 55, l: 55, r: 20 },
          xaxis: {
            title: { text: 'Prey (x)' },
            range: [0, nullclineData.xMax],
          },
          yaxis: {
            title: { text: 'Predator (y)' },
            range: [0, nullclineData.yMax],
          },
          showlegend: true,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>

      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)] font-mono">
          Fixed point: ({fpX.toFixed(1)}, {fpY.toFixed(1)})
        </div>
      </SimulationResults>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          The <span style={{ color: '#ef4444' }}>red dashed line</span> is the prey nullcline (where dx/dt = 0)
          and the <span style={{ color: '#3b82f6' }}>blue dashed line</span> is the predator nullcline (where dy/dt = 0).
          They intersect at the fixed point. All trajectories orbit this point in closed loops &mdash;
          the Lotka-Volterra system conserves a quantity and never spirals inward or outward.
          The grey arrows show the direction of flow at each point in the phase plane.
        </p>
      </div>
    </SimulationPanel>
  );
}
