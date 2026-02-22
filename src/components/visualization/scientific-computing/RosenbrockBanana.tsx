"use client";

import { useState, useMemo, useCallback } from 'react';
import { clamp } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Rosenbrock banana function contour plot with optimizer trajectories.
 * Shows how gradient descent gets stuck in the narrow valley while
 * Newton's method converges quickly.
 */

export default function RosenbrockBanana({}: SimulationComponentProps) {
  const [method, setMethod] = useState<'gd' | 'newton' | 'both'>('both');
  const [lr, setLr] = useState(0.002);
  const [x0, setX0] = useState(-1.0);
  const [y0, setY0] = useState(1.5);
  const [maxSteps, setMaxSteps] = useState(100);

  // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y - x^2)^2
  const rosenbrock = useCallback((x: number, y: number) => {
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2;
  }, []);

  const grad = useCallback((x: number, y: number): [number, number] => {
    const gx = -2 * (1 - x) + 200 * (y - x * x) * (-2 * x);
    const gy = 200 * (y - x * x);
    return [gx, gy];
  }, []);

  const hessian = useCallback((x: number, y: number): [[number, number], [number, number]] => {
    const hxx = 2 + 800 * x * x - 400 * (y - x * x) * 2;
    const hxy = -400 * x;
    const hyy = 200;
    return [[hxx, hxy], [hxy, hyy]];
  }, []);

  // Compute optimizer trajectories
  const gdPath = useMemo(() => {
    if (method !== 'gd' && method !== 'both') return [];
    const path: [number, number][] = [[x0, y0]];
    let px = x0, py = y0;

    for (let i = 0; i < maxSteps; i++) {
      const [gx, gy] = grad(px, py);
      const gnorm = Math.sqrt(gx * gx + gy * gy);
      if (gnorm < 1e-8) break;

      px -= lr * gx;
      py -= lr * gy;

      // Clamp to prevent divergence
      px = clamp(px, -3, 3);
      py = clamp(py, -2, 4);

      path.push([px, py]);
    }
    return path;
  }, [x0, y0, lr, maxSteps, method, grad]);

  const newtonPath = useMemo(() => {
    if (method !== 'newton' && method !== 'both') return [];
    const path: [number, number][] = [[x0, y0]];
    let px = x0, py = y0;

    for (let i = 0; i < Math.min(maxSteps, 50); i++) {
      const [gx, gy] = grad(px, py);
      const gnorm = Math.sqrt(gx * gx + gy * gy);
      if (gnorm < 1e-10) break;

      const [[hxx, hxy], [, hyy]] = hessian(px, py);
      const det = hxx * hyy - hxy * hxy;

      let dx, dy;
      if (Math.abs(det) > 1e-12) {
        dx = (hyy * (-gx) - hxy * (-gy)) / det;
        dy = (hxx * (-gy) - hxy * (-gx)) / det;
      } else {
        // Fall back to gradient descent
        dx = -gx * 0.001;
        dy = -gy * 0.001;
      }

      // Line search with backtracking
      let alpha = 1.0;
      const fCurrent = rosenbrock(px, py);
      for (let ls = 0; ls < 20; ls++) {
        const nx = px + alpha * dx;
        const ny = py + alpha * dy;
        if (rosenbrock(nx, ny) < fCurrent - 1e-4 * alpha * (gx * dx + gy * dy)) {
          break;
        }
        alpha *= 0.5;
      }

      px += alpha * dx;
      py += alpha * dy;

      px = clamp(px, -3, 3);
      py = clamp(py, -2, 4);

      path.push([px, py]);
    }
    return path;
  }, [x0, y0, maxSteps, method, grad, hessian, rosenbrock]);

  // Contour data
  const contourData = useMemo(() => {
    const nx = 120;
    const ny = 120;
    const xmin = -2.5, xmax = 2.5;
    const ymin = -1.5, ymax = 3.5;
    const xgrid = Array.from({ length: nx }, (_, i) => xmin + (xmax - xmin) * i / (nx - 1));
    const ygrid = Array.from({ length: ny }, (_, i) => ymin + (ymax - ymin) * i / (ny - 1));
    const z: number[][] = [];
    for (let iy = 0; iy < ny; iy++) {
      const row: number[] = [];
      for (let ix = 0; ix < nx; ix++) {
        const val = rosenbrock(xgrid[ix], ygrid[iy]);
        // Log scale for better visualization
        row.push(Math.log10(Math.max(val, 1e-4)));
      }
      z.push(row);
    }
    return { xgrid, ygrid, z };
  }, [rosenbrock]);

  // Convergence plot: f(x_k) vs iteration
  const convergenceData = useMemo(() => {
    const gdVals = gdPath.map(([x, y]) => rosenbrock(x, y));
    const newtonVals = newtonPath.map(([x, y]) => rosenbrock(x, y));
    return { gdVals, newtonVals };
  }, [gdPath, newtonPath, rosenbrock]);

  const gdFinal = gdPath.length > 0 ? gdPath[gdPath.length - 1] : [x0, y0];
  const newtonFinal = newtonPath.length > 0 ? newtonPath[newtonPath.length - 1] : [x0, y0];

  return (
    <SimulationPanel title="Rosenbrock Banana Function: Optimizer Comparison" caption="The Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2 has a narrow curved valley. Gradient descent zigzags; Newton's method follows the curvature.">
      <SimulationSettings>
        <div>
          <SimulationLabel>Method</SimulationLabel>
          <SimulationToggle
            options={[
              { label: 'GD', value: 'gd' },
              { label: 'Newton', value: 'newton' },
              { label: 'Both', value: 'both' },
            ]}
            value={method}
            onChange={(v) => setMethod(v as 'gd' | 'newton' | 'both')}
          />
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <SimulationLabel>GD learning rate: {lr.toFixed(4)}</SimulationLabel>
            <Slider
              value={[lr]}
              onValueChange={([v]) => setLr(v)}
              min={0.0001}
              max={0.01}
              step={0.0001}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>x0: {x0.toFixed(1)}</SimulationLabel>
            <Slider
              value={[x0]}
              onValueChange={([v]) => setX0(v)}
              min={-2}
              max={2}
              step={0.1}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>y0: {y0.toFixed(1)}</SimulationLabel>
            <Slider
              value={[y0]}
              onValueChange={([v]) => setY0(v)}
              min={-1}
              max={3}
              step={0.1}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Max steps: {maxSteps}</SimulationLabel>
            <Slider
              value={[maxSteps]}
              onValueChange={([v]) => setMaxSteps(v)}
              min={10}
              max={500}
              step={10}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasHeatmap
            data={[
              {
                z: contourData.z,
                x: contourData.xgrid,
                y: contourData.ygrid,
                colorscale: 'Inferno',
                showscale: true,
              },
            ]}
            layout={{
              title: { text: 'log10(f) Contour + Paths' },
              xaxis: { title: { text: 'x' } },
              yaxis: { title: { text: 'y' } },
              margin: { t: 40, r: 60, b: 50, l: 60 },
            }}
            style={{ width: '100%', height: 420 }}
          />

          <CanvasChart
            data={[
              ...(gdPath.length > 0
                ? [
                    {
                      x: convergenceData.gdVals.map((_, i) => i),
                      y: convergenceData.gdVals,
                      type: 'scatter' as const,
                      mode: 'lines' as const,
                      line: { color: '#3b82f6', width: 2 },
                      name: `GD (${gdPath.length} steps)`,
                    },
                  ]
                : []),
              ...(newtonPath.length > 0
                ? [
                    {
                      x: convergenceData.newtonVals.map((_, i) => i),
                      y: convergenceData.newtonVals,
                      type: 'scatter' as const,
                      mode: 'lines' as const,
                      line: { color: '#ef4444', width: 2 },
                      name: `Newton (${newtonPath.length} steps)`,
                    },
                  ]
                : []),
            ]}
            layout={{
              title: { text: 'Convergence: f(x_k) vs iteration' },
              xaxis: { title: { text: 'Iteration' } },
              yaxis: { title: { text: 'f(x,y)' }, type: 'log' },
              showlegend: true,
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            style={{ width: '100%', height: 420 }}
          />
        </div>
      </SimulationMain>

      <SimulationResults>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">GD final f</div>
            <div className="text-sm font-mono font-semibold text-[#3b82f6]">
              {rosenbrock(gdFinal[0], gdFinal[1]).toExponential(2)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">GD final (x,y)</div>
            <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
              ({gdFinal[0].toFixed(3)}, {gdFinal[1].toFixed(3)})
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Newton final f</div>
            <div className="text-sm font-mono font-semibold text-[#ef4444]">
              {rosenbrock(newtonFinal[0], newtonFinal[1]).toExponential(2)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Newton final (x,y)</div>
            <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
              ({newtonFinal[0].toFixed(3)}, {newtonFinal[1].toFixed(3)})
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
