import React, { useState, useMemo, useCallback } from 'react';
import Plotly from 'react-plotly.js';
import { PlotData } from 'plotly.js';

interface Func {
  name: string;
  f: (x: number) => number;
  df: (x: number) => number;
}

const functions: Func[] = [
  {
    name: 'sin(x)',
    f: (x) => Math.sin(x),
    df: (x) => Math.cos(x),
  },
  {
    name: 'x² - 2',
    f: (x) => x * x - 2,
    df: (x) => 2 * x,
  },
  {
    name: 'eˣ - 2',
    f: (x) => Math.exp(x) - 2,
    df: (x) => Math.exp(x),
  },
  {
    name: '1/(x-3) + 1',
    f: (x) => 1 / (x - 3) + 1,
    df: (x) => -1 / ((x - 3) ** 2),
  },
];

const Newton1D: React.FC = () => {
  const [selectedFunc, setSelectedFunc] = useState(0);
  const [x0, setX0] = useState(1.0);
  const [maxSteps, setMaxSteps] = useState(10);
  const [currentStep, setCurrentStep] = useState(0);
  const [animate, setAnimate] = useState(false);

  const f = functions[selectedFunc].f;
  const df = functions[selectedFunc].df;

  const path = useMemo(() => {
    const path: number[] = [x0];
    let x = x0;
    for (let i = 0; i < maxSteps; i++) {
      const fx = f(x);
      if (Math.abs(fx) < 1e-8) break;
      const dfx = df(x);
      if (Math.abs(dfx) < 1e-12) {
        path.push(x);
        break;
      }
      const dx = -fx / dfx;
      x += dx;
      path.push(x);
    }
    return path;
  }, [f, df, x0, maxSteps]);

  const data = useMemo(() => {
    const xRange = [-5, 5];
    const xGrid = Array.from({length: 1000}, (_, i) => xRange[0] + i * (xRange[1] - xRange[0]) / 999);
    const yGrid = xGrid.map(f);

    const minX = Math.min(...path, xRange[0]);
    const maxX = Math.max(...path, xRange[1]);
    const margin = 0.5;
    const plotX = [minX - margin, maxX + margin];
    const plotY = [-Math.max(...yGrid.map(Math.abs)) -1, Math.max(...yGrid.map(Math.abs)) +1];

    const traces: PlotData[] = [
      {
        x: xGrid,
        y: yGrid,
        type: 'scatter',
        mode: 'lines',
        name: 'f(x)',
        line: {color: 'blue'},
      },
      {
        x: [plotX[0], plotX[1]],
        y: [plotX[0], plotX[1]],
        type: 'scatter',
        mode: 'lines',
        name: 'y = x',
        line: {color: 'black', dash: 'dash'},
      },
    ];

    // Cobweb lines up to currentStep
    for (let i = 0; i < Math.min(currentStep, path.length - 1); i++) {
      const xi = path[i];
      const xi1 = path[i + 1];
      // Horizontal from (xi, xi) to (xi, xi1)
      traces.push({
        x: [xi, xi],
        y: [xi, xi1],
        type: 'scatter',
        mode: 'lines',
        line: {color: 'red', width: 3},
        showlegend: false,
      });
      // Vertical from (xi, xi1) to (xi1, xi1)
      traces.push({
        x: [xi, xi1],
        y: [xi1, xi1],
        type: 'scatter',
        mode: 'lines',
        line: {color: 'red', width: 3},
        showlegend: false,
      });
    }

    // Current point
    if (path.length > 0) {
      traces.push({
        x: [path[0]],
        y: [path[0]],
        type: 'scatter',
        mode: 'markers',
        marker: {color: 'green', size: 12},
        name: 'Start',
      });
    }
    if (currentStep < path.length) {
      traces.push({
        x: [path[currentStep]],
        y: [path[currentStep]],
        type: 'scatter',
        mode: 'markers',
        marker: {color: 'orange', size: 10},
        name: 'Current',
      });
    }

    return traces;
  }, [f, path, currentStep]);

  const layout = {
    title: `${functions[selectedFunc].name} - Newton's Method Cobweb`,
    xaxis: {title: 'x'},
    yaxis: {title: 'y'},
    width: 800,
    height: 600,
  };

  const handleStep = () => {
    setCurrentStep((prev) => Math.min(prev + 1, path.length - 1));
  };

  const handleReset = () => {
    setCurrentStep(0);
  };

  React.useEffect(() => {
    setCurrentStep(0);
  }, [x0, selectedFunc, maxSteps]);

  return (
    <div className="p-4">
      <div className="grid grid-cols-4 gap-4 mb-4">
        <div>
          <label>Function:</label>
          <select value={selectedFunc} onChange={(e) => setSelectedFunc(Number(e.target.value))} className="ml-2">
            {functions.map((func, idx) => (
              <option key={idx} value={idx}>{func.name}</option>
            ))}
          </select>
        </div>
        <div>
          <label>x₀:</label>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.1}
            value={x0}
            onChange={(e) => setX0(Number(e.target.value))}
            className="ml-2"
          />
          <span>{x0.toFixed(2)}</span>
        </div>
        <div>
          <label>Max Steps:</label>
          <input
            type="range"
            min={5}
            max={20}
            value={maxSteps}
            onChange={(e) => setMaxSteps(Number(e.target.value))}
            className="ml-2"
          />
          <span>{maxSteps}</span>
        </div>
        <div>
          <button onClick={handleStep} className="bg-blue-500 text-white px-4 py-1 mr-2 rounded">Step</button>
          <button onClick={handleReset} className="bg-gray-500 text-white px-4 py-1 mr-2 rounded">Reset</button>
        </div>
      </div>
      <div>Path length: {path.length}, Current step: {currentStep}, Final x: {path[path.length-1]?.toFixed(6)}</div>
      <Plotly data={data} layout={layout} />
    </div>
  );
};

export default Newton1D;
