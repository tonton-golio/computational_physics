"use client";

import { useState, useMemo } from 'react';
import { mulberry32, clamp } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


// Cost function: f(x,y) = a*sin(x*b) + c*cos(y*d) + e*x + c1*y + x^4/100 + (y-1)^4/100 + y/4 + 2
function costFunction(x: number, y: number, params: number[]): number {
  const [a, b, c, d, e, c1] = params;
  return a * Math.sin(x * b) + c * Math.cos(y * d) + e * x + c1 * y +
    (1 / 100) * Math.pow(x, 4) + (1 / 100) * Math.pow(y - 1, 4) + y / 4 + 2;
}

const COST_PARAMS = [1, 1, 0.77, 1.11, 0.31, 0.31];

export default function SteepestDescent({}: SimulationComponentProps) {
  const [lr, setLr] = useState(0.1);
  const [nSteps, setNSteps] = useState(30);
  const [seed, setSeed] = useState(1);

  const result = useMemo(() => {
    // Generate contour grid
    const nGrid = 50;
    const low = -4, high = 4;
    const step = (high - low) / (nGrid - 1);
    const xArr: number[] = [];
    const yArr: number[] = [];
    for (let i = 0; i < nGrid; i++) {
      xArr.push(low + i * step);
      yArr.push(low + i * step);
    }

    const zGrid: number[][] = [];
    for (let j = 0; j < nGrid; j++) {
      const row: number[] = [];
      for (let i = 0; i < nGrid; i++) {
        row.push(costFunction(xArr[i], yArr[j], COST_PARAMS));
      }
      zGrid.push(row);
    }

    // Start from a seeded pseudo-random point
    const rng = mulberry32(seed);
    const x0 = -3 + 6 * rng();
    const y0 = -3 + 6 * rng();

    // Steepest descent
    const h = 0.001;
    const pathX: number[] = [x0];
    const pathY: number[] = [y0];
    const pathZ: number[] = [costFunction(x0, y0, COST_PARAMS)];

    let cx = x0;
    let cy = y0;
    for (let i = 0; i < nSteps; i++) {
      const fCurrent = costFunction(cx, cy, COST_PARAMS);
      const gradX = (costFunction(cx + h, cy, COST_PARAMS) - fCurrent) / h;
      const gradY = (costFunction(cx, cy + h, COST_PARAMS) - fCurrent) / h;

      cx -= gradX * lr;
      cy -= gradY * lr;

      // Clamp to grid bounds
      cx = clamp(cx, low, high);
      cy = clamp(cy, low, high);

      pathX.push(cx);
      pathY.push(cy);
      pathZ.push(costFunction(cx, cy, COST_PARAMS));
    }

    return { xArr, yArr, zGrid, pathX, pathY, pathZ, x0, y0 };
  }, [lr, nSteps, seed]);

  const lrOptions = [0.001, 0.003, 0.01, 0.032, 0.1, 0.316, 1.0];

  return (
    <SimulationPanel title="Steepest Descent Optimization" caption="Gradient descent on a non-convex surface. The optimizer follows the negative gradient with a given learning rate. Observe how step size and number of iterations affect convergence.">
      <SimulationSettings>
        <div>
          <SimulationLabel className="text-[var(--text-muted)] text-sm">Learning rate: {lr}</SimulationLabel>
          <select
            value={lr}
            onChange={(e) => setLr(parseFloat(e.target.value))}
            className="w-full bg-[var(--surface-2)] text-[var(--text-strong)] border border-[var(--border-strong)] rounded p-1 text-sm mt-1"
          >
            {lrOptions.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-sm">Steps: {nSteps}</SimulationLabel>
            <Slider
              min={1} max={100} step={1} value={[nSteps]}
              onValueChange={([v]) => setNSteps(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-sm">Random start (seed): {seed}</SimulationLabel>
            <Slider
              min={1} max={20} step={1} value={[seed]}
              onValueChange={([v]) => setSeed(v)}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="relative">
          <CanvasHeatmap
            data={[{
              z: result.zGrid,
              x: result.xArr,
              y: result.yArr,
              type: 'heatmap',
              colorscale: 'Inferno',
            }]}
            layout={{
              title: { text: 'Contour Plot with Descent Path' },
              xaxis: { title: { text: 'x' } },
              yaxis: { title: { text: 'y' } },
              height: 450,
              legend: { x: 0.02, y: 0.98 },
              margin: { t: 40, b: 50, l: 50, r: 20 },
            }}
            style={{ width: '100%' }}
          />
          <CanvasChart
            data={[
              {
                x: result.pathX,
                y: result.pathY,
                type: 'scatter' as const,
                mode: 'lines+markers' as const,
                marker: { color: '#22d3ee', size: 4 },
                line: { color: '#22d3ee', width: 2 },
                name: 'Descent path',
              },
              {
                x: [result.x0],
                y: [result.pathY[0]],
                type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { color: '#fbbf24', size: 10, symbol: 'star' },
                name: 'Start',
              },
              {
                x: [result.pathX[result.pathX.length - 1]],
                y: [result.pathY[result.pathY.length - 1]],
                type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { color: '#34d399', size: 10, symbol: 'diamond' },
                name: 'End',
              },
            ]}
            layout={{
              xaxis: { range: [-4, 4] },
              yaxis: { range: [-4, 4] },
              height: 450,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              showlegend: true,
              legend: { x: 0.02, y: 0.98 },
              margin: { t: 40, b: 50, l: 50, r: 20 },
            }}
            style={{ width: '100%', position: 'absolute', top: 0, left: 0 }}
          />
        </div>
        <CanvasChart
          data={[{
            y: result.pathZ,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            marker: { color: '#f472b6', size: 4 },
            line: { color: '#f472b6' },
            name: 'Cost f(x,y)',
          }]}
          layout={{
            title: { text: 'Cost Function vs Iteration' },
            xaxis: { title: { text: 'Step' } },
            yaxis: { title: { text: 'f(x, y)' } },
            height: 450,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
      </div>
      <p className="text-[var(--text-soft)] text-xs mt-3">
        Try different starting seeds and learning rates to see how the optimizer can get stuck in local minima or overshoot.
      </p>
      </SimulationMain>
    </SimulationPanel>
  );
}
