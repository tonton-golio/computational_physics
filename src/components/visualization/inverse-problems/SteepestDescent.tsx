'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

// Cost function: f(x,y) = a*sin(x*b) + c*cos(y*d) + e*x + c1*y + x^4/100 + (y-1)^4/100 + y/4 + 2
function costFunction(x: number, y: number, params: number[]): number {
  const [a, b, c, d, e, c1] = params;
  return a * Math.sin(x * b) + c * Math.cos(y * d) + e * x + c1 * y +
    (1 / 100) * Math.pow(x, 4) + (1 / 100) * Math.pow(y - 1, 4) + y / 4 + 2;
}

export default function SteepestDescent({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [lr, setLr] = useState(0.1);
  const [nSteps, setNSteps] = useState(30);
  const [seed, setSeed] = useState(1);

  const params = [1, 1, 0.77, 1.11, 0.31, 0.31];

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
        row.push(costFunction(xArr[i], yArr[j], params));
      }
      zGrid.push(row);
    }

    // Start from a seeded pseudo-random point
    const pseudoRng = (s: number) => {
      const v = Math.sin(s * 12.9898 + 78.233) * 43758.5453;
      return v - Math.floor(v);
    };
    const x0 = -3 + 6 * pseudoRng(seed);
    const y0 = -3 + 6 * pseudoRng(seed + 0.5);

    // Steepest descent
    const h = 0.001;
    const pathX: number[] = [x0];
    const pathY: number[] = [y0];
    const pathZ: number[] = [costFunction(x0, y0, params)];

    let cx = x0;
    let cy = y0;
    for (let i = 0; i < nSteps; i++) {
      const fCurrent = costFunction(cx, cy, params);
      const gradX = (costFunction(cx + h, cy, params) - fCurrent) / h;
      const gradY = (costFunction(cx, cy + h, params) - fCurrent) / h;

      cx -= gradX * lr;
      cy -= gradY * lr;

      // Clamp to grid bounds
      cx = Math.max(low, Math.min(high, cx));
      cy = Math.max(low, Math.min(high, cy));

      pathX.push(cx);
      pathY.push(cy);
      pathZ.push(costFunction(cx, cy, params));
    }

    return { xArr, yArr, zGrid, pathX, pathY, pathZ, x0, y0 };
  }, [lr, nSteps, seed]);

  const lrOptions = [0.001, 0.003, 0.01, 0.032, 0.1, 0.316, 1.0];

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Steepest Descent Optimization</h3>
      <p className="text-gray-400 text-sm mb-4">
        Gradient descent on a non-convex surface. The optimizer follows the negative gradient with a
        given learning rate. Observe how step size and number of iterations affect convergence.
      </p>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-gray-300 text-sm">Learning rate: {lr}</label>
          <select
            value={lr}
            onChange={(e) => setLr(parseFloat(e.target.value))}
            className="w-full bg-[#0a0a15] text-white border border-gray-700 rounded p-1 text-sm mt-1"
          >
            {lrOptions.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-gray-300 text-sm">Steps: {nSteps}</label>
          <input
            type="range" min={1} max={100} step={1} value={nSteps}
            onChange={(e) => setNSteps(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-gray-300 text-sm">Random start (seed): {seed}</label>
          <input
            type="range" min={1} max={20} step={1} value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Plot
          data={[
            {
              x: result.xArr,
              y: result.yArr,
              z: result.zGrid,
              type: 'contour' as const,
              colorscale: 'Inferno',
              ncontours: 30,
              showscale: true,
              colorbar: { tickfont: { color: '#9ca3af' } },
              contours: { coloring: 'heatmap' },
            },
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
            title: { text: 'Contour Plot with Descent Path' },
            xaxis: { title: { text: 'x' }, color: '#9ca3af' },
            yaxis: { title: { text: 'y' }, color: '#9ca3af' },
            height: 450,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(0,0,0,0.3)' },
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
        <Plot
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
            xaxis: { title: { text: 'Step' }, color: '#9ca3af' },
            yaxis: { title: { text: 'f(x, y)' }, color: '#9ca3af' },
            height: 450,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>
      <p className="text-gray-500 text-xs mt-3">
        Try different starting seeds and learning rates to see how the optimizer can get stuck in local minima or overshoot.
      </p>
    </div>
  );
}
