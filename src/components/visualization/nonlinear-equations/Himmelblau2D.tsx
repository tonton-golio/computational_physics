import React, { useState, useMemo, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const Himmelblau2D: React.FC<SimulationComponentProps> = () => {
  const [method, setMethod] = useState<'gd' | 'newton'>('gd');
  const [lr, setLr] = useState(0.01);
  const [x0, setX0] = useState(-1.0);
  const [y0, setY0] = useState(1.5);
  const [maxSteps, setMaxSteps] = useState(20);
  const [currentStep, setCurrentStep] = useState(0);
  const [showBasins, setShowBasins] = useState(false);

  // Himmelblau function
  const f = useCallback((x: number, y: number): number => {
    const u = x * x + y - 11;
    const v = x + y * y - 7;
    return u * u + v * v;
  }, []);

  const gradX = useCallback((x: number, y: number): number => {
    const u = x * x + y - 11;
    const v = x + y * y - 7;
    return 4 * x * u + 2 * v;
  }, []);

  const gradY = useCallback((x: number, y: number): number => {
    const u = x * x + y - 11;
    const v = x + y * y - 7;
    return 2 * u + 4 * y * v;
  }, []);

  const hessXX = useCallback((x: number, y: number): number => {
    return 12 * x * x + 4 * y - 42;
  }, []);

  const hessXY = useCallback((x: number, y: number): number => {
    return 4 * x + 4 * y;
  }, []);

  const hessYY = useCallback((x: number, y: number): number => {
    return 4 * x + 12 * y * y - 26;
  }, []);

  const path = useMemo(() => {
    const path: [number, number][] = [[x0, y0]];
    let px = x0;
    let py = y0;
    for (let i = 0; i < maxSteps; i++) {
      const gx = gradX(px, py);
      const gy = gradY(px, py);
      const gnorm = Math.sqrt(gx * gx + gy * gy);
      if (gnorm < 1e-6) break;

      let dx = 0;
      let dy = 0;

      if (method === 'newton') {
        const hxx = hessXX(px, py);
        const hxy = hessXY(px, py);
        const hyy = hessYY(px, py);
        const det = hxx * hyy - hxy * hxy;
        if (Math.abs(det) > 1e-12) {
          dx = (hyy * (-gx) - hxy * (-gy)) / det;
          dy = (hxx * (-gy) - hxy * (-gx)) / det;
        } else {
          dx = -gx * lr;
          dy = -gy * lr;
        }
      } else { // gd
        dx = -gx * lr;
        dy = -gy * lr;
      }

      px += dx;
      py += dy;
      path.push([px, py]);
    }
    return path;
  }, [x0, y0, method, lr, maxSteps, gradX, gradY, hessXX, hessXY, hessYY]);

  // Contour data
  const contourData = useMemo(() => {
    const nx = 100;
    const ny = 100;
    const xmin = -6;
    const xmax = 6;
    const ymin = -6;
    const ymax = 6;
    const xgrid = new Array(nx).fill(0).map((_, i) => xmin + (xmax - xmin) * i / (nx - 1));
    const ygrid = new Array(ny).fill(0).map((_, i) => ymin + (ymax - ymin) * i / (ny - 1));
    const z: number[][] = [];
    for (let iy = 0; iy < ny; iy++) {
      const row: number[] = [];
      for (let ix = 0; ix < nx; ix++) {
        row.push(f(xgrid[ix], ygrid[iy]));
      }
      z.push(row);
    }
    return { xgrid, ygrid, z };
  }, [f]);

  // Path trace
  const pathTrace = useMemo(() => ({
    x: path.slice(0, currentStep + 1).map(p => p[0]),
    y: path.slice(0, currentStep + 1).map(p => p[1]),
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    line: {color: '#ef4444', width: 3},
    marker: {size: 8, color: '#ef4444'},
    name: 'Descent path',
  }), [path, currentStep]);

  // Basins
  const attractors = useMemo(() => [
    [3.0, 2.0],
    [-2.805, 3.131],
    [-3.779, -3.283],
    [3.584, -1.848],
  ], []);

  const basinData = useMemo(() => {
    if (!showBasins) return null;
    const nx = 40;
    const ny = 40;
    const xmin = -5;
    const xmax = 5;
    const ymin = -5;
    const ymax = 5;
    const xgrid = new Array(nx).fill(0).map((_, i) => xmin + (xmax - xmin) * i / (nx - 1));
    const ygrid = new Array(ny).fill(0).map((_, i) => ymin + (ymax - ymin) * i / (ny - 1));
    const zbasin: number[][] = [];
    for (let iy = 0; iy < ny; iy++) {
      const row: number[] = [];
      for (let ix = 0; ix < nx; ix++) {
        // Run optimization from init xgrid[ix], ygrid[iy]
        let px = xgrid[ix];
        let py = ygrid[iy];
        for (let it = 0; it < 50; it++) {
          const gx = gradX(px, py);
          const gy = gradY(px, py);
          const gnorm = Math.sqrt(gx**2 + gy**2);
          if (gnorm < 1e-4) break;
          let dx = 0, dy = 0;
          if (method === 'newton') {
            const hxx = hessXX(px, py);
            const hxy = hessXY(px, py);
            const hyy = hessYY(px, py);
            const det = hxx * hyy - hxy * hxy;
            if (Math.abs(det) > 1e-8) {
              dx = (hyy * (-gx) - hxy * (-gy)) / det;
              dy = (hxx * (-gy) - hxy * (-gx)) / det;
            } else {
              dx = -gx * 0.01;
              dy = -gy * 0.01;
            }
          } else {
            dx = -gx * 0.01;
            dy = -gy * 0.01;
          }
          px += dx;
          py += dy;
        }
        // Find closest attractor
        let minDist = Infinity;
        let closest = 0;
        for (let a = 0; a < attractors.length; a++) {
          const dx = px - attractors[a][0];
          const dy = py - attractors[a][1];
          const dist = Math.sqrt(dx*dx + dy*dy);
          if (dist < minDist) {
            minDist = dist;
            closest = a;
          }
        }
        row.push(closest);
      }
      zbasin.push(row);
    }
    return { xgrid, ygrid, zbasin };
  }, [showBasins, method, gradX, gradY, hessXX, hessXY, hessYY, attractors]);

  const handleStep = () => {
    setCurrentStep((prev) => Math.min(prev + 1, path.length - 1));
  };

  const handleReset = () => {
    setCurrentStep(0);
  };

  React.useEffect(() => {
    setCurrentStep(0);
  }, [x0, y0, method, lr]);

  return (
    <SimulationPanel title="Himmelblau Function Optimization">
      <SimulationSettings>
        <div>
          <SimulationLabel>Method:</SimulationLabel>
          <select value={method} onChange={(e) => setMethod(e.target.value as 'gd' | 'newton')} className="ml-2">
            <option value="gd">Gradient Descent</option>
            <option value="newton">Newton</option>
          </select>
        </div>
        <div className="flex items-center">
          <SimulationButton variant="primary" onClick={() => setShowBasins(!showBasins)}>
            {showBasins ? 'Hide' : 'Show'} Basins
          </SimulationButton>
          <SimulationButton variant="primary" onClick={handleStep}>Step</SimulationButton>
          <SimulationButton onClick={handleReset}>Reset</SimulationButton>
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>LR: {lr.toFixed(3)}</SimulationLabel>
          <Slider min={0.001} max={0.1} step={0.001} value={[lr]} onValueChange={([v]) => setLr(v)} className="ml-2 w-32" />
        </div>
        <div>
          <SimulationLabel>x₀: {x0.toFixed(1)}</SimulationLabel>
          <Slider min={-5} max={5} step={0.1} value={[x0]} onValueChange={([v]) => setX0(v)} className="ml-2" />
        </div>
        <div>
          <SimulationLabel>y₀: {y0.toFixed(1)}</SimulationLabel>
          <Slider min={-5} max={5} step={0.1} value={[y0]} onValueChange={([v]) => setY0(v)} className="ml-2" />
        </div>
        <div>
          <SimulationLabel>Steps: {maxSteps}</SimulationLabel>
          <Slider min={10} max={50} value={[maxSteps]} onValueChange={([v]) => setMaxSteps(v)} className="ml-2" />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {showBasins && basinData ? (
            <CanvasHeatmap
              data={[{ z: basinData.zbasin, x: basinData.xgrid, y: basinData.ygrid, colorscale: 'Portland' }]}
              layout={{ title: { text: 'Basins of Attraction' }, xaxis: { title: { text: 'x' } }, yaxis: { title: { text: 'y' } } }}
              style={{ width: '100%', height: 450 }}
            />
          ) : (
            <CanvasHeatmap
              data={[{ z: contourData.z, x: contourData.xgrid, y: contourData.ygrid, colorscale: 'Viridis' }]}
              layout={{ title: { text: 'Himmelblau f(x,y)' }, xaxis: { title: { text: 'x' } }, yaxis: { title: { text: 'y' } } }}
              style={{ width: '100%', height: 450 }}
            />
          )}
          <CanvasChart
            data={[
              pathTrace,
              {
                x: [path[0]?.[0] ?? 0],
                y: [path[0]?.[1] ?? 0],
                type: 'scatter' as const,
                mode: 'markers' as const,
                marker: { size: 12, color: '#22c55e', symbol: 'diamond' },
                name: 'Start',
              },
            ]}
            layout={{
              title: { text: `${method.toUpperCase()} Descent Path` },
              xaxis: { title: { text: 'x' }, range: [-6, 6] },
              yaxis: { title: { text: 'y' }, range: [-6, 6] },
            }}
            style={{ width: '100%', height: 450 }}
          />
        </div>
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">Final: ({path[path.length-1]?.[0]?.toFixed(3) || 0}, {path[path.length-1]?.[1]?.toFixed(3) || 0}), f={f(path[path.length-1]?.[0] || 0, path[path.length-1]?.[1] || 0)?.toFixed(2)}</div>
      </SimulationResults>
    </SimulationPanel>
  );
};

export default Himmelblau2D;
