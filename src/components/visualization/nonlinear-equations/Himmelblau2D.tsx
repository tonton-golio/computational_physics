import React, { useState, useMemo, useCallback } from 'react';
import Plotly from 'react-plotly.js';

const Himmelblau2D: React.FC = () => {
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
    return [{
      type: 'contour',
      x: xgrid,
      y: ygrid,
      z,
      colorscale: 'Viridis',
      contours: { coloring: 'heatmap' },
      name: 'Himmelblau f(x,y)',
    }];
  }, [f]);

  // Path trace
  const pathTrace = useMemo(() => ({
    x: path.slice(0, currentStep + 1).map(p => p[0]),
    y: path.slice(0, currentStep + 1).map(p => p[1]),
    type: 'scatter',
    mode: 'lines+markers',
    line: {color: 'white', width: 4},
    marker: {size: 8, color: 'red'},
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
    if (!showBasins) return [];
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
    return [{
      type: 'heatmap',
      x: xgrid,
      y: ygrid,
      z: zbasin,
      colorscale: 'RdYlBu',
      colorbar: {title: 'Basin'},
      name: 'Basins of attraction',
      opacity: 0.6,
    }];
  }, [showBasins, method, gradX, gradY, hessXX, hessXY, hessYY, attractors]);

  const data = [...contourData, pathTrace, ...basinData];

  const layout = {
    title: `Himmelblau Function - ${method.toUpperCase()} Descent & Basins`,
    xaxis: {title: 'x'},
    yaxis: {title: 'y'},
    width: 800,
    height: 700,
  };

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
    <div className="p-4">
      <div className="grid grid-cols-6 gap-4 mb-4">
        <div>
          <label>Method:</label>
          <select value={method} onChange={(e) => setMethod(e.target.value as 'gd' | 'newton')} className="ml-2">
            <option value="gd">Gradient Descent</option>
            <option value="newton">Newton</option>
          </select>
        </div>
        <div>
          <label>LR:</label>
          <input type="range" min={0.001} max={0.1} step={0.001} value={lr} onChange={(e) => setLr(Number(e.target.value))} className="ml-2 w-32" />
          <span>{lr.toFixed(3)}</span>
        </div>
        <div>
          <label>x₀:</label>
          <input type="range" min={-5} max={5} step={0.1} value={x0} onChange={(e) => setX0(Number(e.target.value))} className="ml-2" />
          <span>{x0.toFixed(1)}</span>
        </div>
        <div>
          <label>y₀:</label>
          <input type="range" min={-5} max={5} step={0.1} value={y0} onChange={(e) => setY0(Number(e.target.value))} className="ml-2" />
          <span>{y0.toFixed(1)}</span>
        </div>
        <div>
          <label>Steps:</label>
          <input type="range" min={10} max={50} value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} className="ml-2" />
          <span>{maxSteps}</span>
        </div>
        <div className="flex items-center">
          <button onClick={() => setShowBasins(!showBasins)} className="bg-green-500 text-white px-4 py-1 rounded mr-2">
            {showBasins ? 'Hide' : 'Show'} Basins
          </button>
          <button onClick={handleStep} className="bg-blue-500 text-white px-4 py-1 rounded mr-2">Step</button>
          <button onClick={handleReset} className="bg-gray-500 text-white px-4 py-1 rounded">Reset</button>
        </div>
      </div>
      <div>Final: ({path[path.length-1]?.[0]?.toFixed(3) || 0}, {path[path.length-1]?.[1]?.toFixed(3) || 0}), f={f(path[path.length-1]?.[0] || 0, path[path.length-1]?.[1] || 0)?.toFixed(2)}</div>
      <Plotly data={data} layout={layout} />
    </div>
  );
};

export default Himmelblau2D;
