"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * 2D elliptical quadratic: f(x,y) = 0.5 * (a*x^2 + b*y^2)
 * where a=conditionNumber, b=1. Minimum at origin.
 */

export default function ConjugateGradientRace({}: SimulationComponentProps) {
  const [condNum, setCondNum] = useState(10);
  const [startX, setStartX] = useState(4);
  const [startY, setStartY] = useState(3);

  const result = useMemo(() => {
    const a = condNum;
    const b = 1;

    // The system is A*x = 0, where A = diag(a, b).
    // We solve min 0.5 * x^T A x (the RHS is zero for minimum at origin).
    // But to make it more interesting, let's solve A*x = rhs for a given rhs.
    // Actually, the standard demo: minimize f(x) = 0.5*x^T*A*x - b^T*x
    // Let's keep it simple: minimize 0.5*(a*x^2 + y^2), minimum at (0,0).
    // SD and CG both start from (startX, startY).

    const cost = (x: number, y: number) => 0.5 * (a * x * x + b * y * y);

    // Generate contour grid
    const nGrid = 60;
    const lo = -5, hi = 5;
    const step = (hi - lo) / (nGrid - 1);
    const xArr: number[] = [];
    const yArr: number[] = [];
    for (let i = 0; i < nGrid; i++) {
      xArr.push(lo + i * step);
      yArr.push(lo + i * step);
    }
    const zGrid: number[][] = [];
    for (let j = 0; j < nGrid; j++) {
      const row: number[] = [];
      for (let i = 0; i < nGrid; i++) {
        row.push(cost(xArr[i], yArr[j]));
      }
      zGrid.push(row);
    }

    // --- Steepest Descent ---
    const maxSteps = 100;
    const sdX: number[] = [startX];
    const sdY: number[] = [startY];
    const sdCost: number[] = [cost(startX, startY)];

    let sx = startX, sy = startY;
    for (let k = 0; k < maxSteps; k++) {
      // gradient of 0.5*(a*x^2 + b*y^2) is (a*x, b*y)
      const gx = a * sx;
      const gy = b * sy;
      const gnorm2 = gx * gx + gy * gy;
      if (gnorm2 < 1e-12) break;
      // Exact line search: alpha = g^T g / (g^T A g)
      const gAg = a * gx * gx + b * gy * gy;
      const alpha = gnorm2 / gAg;
      sx -= alpha * gx;
      sy -= alpha * gy;
      sdX.push(sx);
      sdY.push(sy);
      sdCost.push(cost(sx, sy));
      if (cost(sx, sy) < 1e-10) break;
    }

    // --- Conjugate Gradients ---
    // For A = diag(a,b), solving A*x = 0 from start point
    // CG on quadratic: r0 = -grad = -(a*x0, b*y0), d0 = r0
    const cgX: number[] = [startX];
    const cgY: number[] = [startY];
    const cgCost: number[] = [cost(startX, startY)];

    let cx = startX, cy = startY;
    let rx = -(a * cx), ry = -(b * cy); // r = -gradient = b - Ax (here b=0)
    let dx = rx, dy = ry;

    for (let k = 0; k < maxSteps; k++) {
      const rnorm2 = rx * rx + ry * ry;
      if (rnorm2 < 1e-12) break;
      // A*d
      const adx = a * dx;
      const ady = b * dy;
      const dAd = dx * adx + dy * ady;
      const alphaCG = rnorm2 / dAd;
      cx += alphaCG * dx;
      cy += alphaCG * dy;
      cgX.push(cx);
      cgY.push(cy);
      cgCost.push(cost(cx, cy));
      // new residual
      const rx_new = rx - alphaCG * adx;
      const ry_new = ry - alphaCG * ady;
      const rnorm2_new = rx_new * rx_new + ry_new * ry_new;
      if (rnorm2_new < 1e-12) break;
      const beta = rnorm2_new / rnorm2;
      dx = rx_new + beta * dx;
      dy = ry_new + beta * dy;
      rx = rx_new;
      ry = ry_new;
    }

    return {
      xArr, yArr, zGrid,
      sdX, sdY, sdCost,
      cgX, cgY, cgCost,
    };
  }, [condNum, startX, startY]);

  const sdSteps = result.sdCost.length - 1;
  const cgSteps = result.cgCost.length - 1;
  const sdFinal = result.sdCost[result.sdCost.length - 1];
  const cgFinal = result.cgCost[result.cgCost.length - 1];

  // Build iteration axis for cost chart
  const maxIter = Math.max(result.sdCost.length, result.cgCost.length);
  const iterAxis = Array.from({ length: maxIter }, (_, i) => i);

  return (
    <SimulationPanel title="Conjugate Gradient vs Steepest Descent" caption="Both methods minimize a 2D elliptical quadratic with adjustable condition number. CG (green) converges in at most 2 steps for a 2D quadratic; SD (cyan) zigzags in narrow valleys.">
      <SimulationConfig>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-sm">Condition number: {condNum}</SimulationLabel>
            <Slider
              min={1} max={100} step={1} value={[condNum]}
              onValueChange={([v]) => setCondNum(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-sm">Start x: {startX.toFixed(1)}</SimulationLabel>
            <Slider
              min={-4.5} max={4.5} step={0.1} value={[startX]}
              onValueChange={([v]) => setStartX(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-sm">Start y: {startY.toFixed(1)}</SimulationLabel>
            <Slider
              min={-4.5} max={4.5} step={0.1} value={[startY]}
              onValueChange={([v]) => setStartY(v)}
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
              title: { text: 'Contour with Optimization Paths' },
              xaxis: { title: { text: 'x' } },
              yaxis: { title: { text: 'y' } },
              height: 450,
              margin: { t: 40, b: 50, l: 50, r: 20 },
            }}
            style={{ width: '100%' }}
          />
          <CanvasChart
            data={[
              {
                x: result.sdX,
                y: result.sdY,
                type: 'scatter' as const,
                mode: 'lines+markers' as const,
                marker: { color: '#22d3ee', size: 4 },
                line: { color: '#22d3ee', width: 2 },
                name: 'Steepest Descent',
              },
              {
                x: result.cgX,
                y: result.cgY,
                type: 'scatter' as const,
                mode: 'lines+markers' as const,
                marker: { color: '#4ade80', size: 5 },
                line: { color: '#4ade80', width: 2.5 },
                name: 'Conjugate Gradients',
              },
              {
                x: [startX],
                y: [startY],
                type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { color: '#fbbf24', size: 10, symbol: 'star' },
                name: 'Start',
              },
              {
                x: [0],
                y: [0],
                type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { color: '#f472b6', size: 8, symbol: 'diamond' },
                name: 'Minimum',
              },
            ]}
            layout={{
              xaxis: { range: [-5, 5] },
              yaxis: { range: [-5, 5] },
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
          data={[
            {
              x: iterAxis.slice(0, result.sdCost.length),
              y: result.sdCost,
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              marker: { color: '#22d3ee', size: 4 },
              line: { color: '#22d3ee', width: 2 },
              name: 'SD cost',
            },
            {
              x: iterAxis.slice(0, result.cgCost.length),
              y: result.cgCost,
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              marker: { color: '#4ade80', size: 5 },
              line: { color: '#4ade80', width: 2.5 },
              name: 'CG cost',
            },
          ]}
          layout={{
            title: { text: 'Cost vs Iteration' },
            xaxis: { title: { text: 'Iteration' } },
            yaxis: { title: { text: 'f(x, y)' }, type: 'log' },
            height: 450,
            showlegend: true,
            legend: { x: 0.6, y: 0.98 },
            margin: { t: 40, b: 50, l: 60, r: 20 },
          }}
          style={{ width: '100%' }}
        />
      </div>
      <p className="text-[var(--text-soft)] text-xs mt-2">
        For a 2D quadratic, CG converges in exactly 2 steps regardless of condition number.
        SD zigzags more as the condition number increases.
      </p>
      </SimulationMain>
      <SimulationResults>
      <div className="mt-3 grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
        <div className="text-[var(--text-muted)]">
          SD steps: <span className="text-[#22d3ee] font-mono">{sdSteps}</span>
        </div>
        <div className="text-[var(--text-muted)]">
          SD final cost: <span className="text-[#22d3ee] font-mono">{sdFinal.toExponential(2)}</span>
        </div>
        <div className="text-[var(--text-muted)]">
          CG steps: <span className="text-[#4ade80] font-mono">{cgSteps}</span>
        </div>
        <div className="text-[var(--text-muted)]">
          CG final cost: <span className="text-[#4ade80] font-mono">{cgFinal.toExponential(2)}</span>
        </div>
      </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
