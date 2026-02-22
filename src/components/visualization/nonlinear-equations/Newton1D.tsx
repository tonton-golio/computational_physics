import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

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

const Newton1D: React.FC<SimulationComponentProps> = () => {
  const [selectedFunc, setSelectedFunc] = useState(0);
  const [x0, setX0] = useState(1.0);
  const [maxSteps, setMaxSteps] = useState(10);
  const [currentStep, setCurrentStep] = useState(0);

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

    const traces: any[] = [
      {
        x: xGrid,
        y: yGrid,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'f(x)',
        line: {color: 'blue'},
      },
      {
        x: [plotX[0], plotX[1]],
        y: [plotX[0], plotX[1]],
        type: 'scatter' as const,
        mode: 'lines' as const,
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
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: {color: 'red', width: 3},
        showlegend: false,
      });
      // Vertical from (xi, xi1) to (xi1, xi1)
      traces.push({
        x: [xi, xi1],
        y: [xi1, xi1],
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: {color: 'red', width: 3},
        showlegend: false,
      });
    }

    // Current point
    if (path.length > 0) {
      traces.push({
        x: [path[0]],
        y: [path[0]],
        type: 'scatter' as const,
        mode: 'markers' as const,
        marker: {color: 'green', size: 12},
        name: 'Start',
      });
    }
    if (currentStep < path.length) {
      traces.push({
        x: [path[currentStep]],
        y: [path[currentStep]],
        type: 'scatter' as const,
        mode: 'markers' as const,
        marker: {color: 'orange', size: 10},
        name: 'Current',
      });
    }

    return traces;
  }, [f, path, currentStep]);

  const layout = {
    title: { text: `${functions[selectedFunc].name} - Newton's Method Cobweb` },
    xaxis: {title: { text: 'x' }},
    yaxis: {title: { text: 'y' }},
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
    <SimulationPanel title="Newton's Method (1D)">
      <SimulationSettings>
        <div>
          <SimulationLabel>Function:</SimulationLabel>
          <select value={selectedFunc} onChange={(e) => setSelectedFunc(Number(e.target.value))} className="ml-2">
            {functions.map((func, idx) => (
              <option key={idx} value={idx}>{func.name}</option>
            ))}
          </select>
        </div>
        <div>
          <SimulationButton variant="primary" onClick={handleStep}>Step</SimulationButton>
          <SimulationButton onClick={handleReset}>Reset</SimulationButton>
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>x₀: {x0.toFixed(2)}</SimulationLabel>
          <Slider
            min={-5}
            max={5}
            step={0.1}
            value={[x0]}
            onValueChange={([v]) => setX0(v)}
            className="ml-2"
          />
        </div>
        <div>
          <SimulationLabel>Max Steps: {maxSteps}</SimulationLabel>
          <Slider
            min={5}
            max={20}
            value={[maxSteps]}
            onValueChange={([v]) => setMaxSteps(v)}
            className="ml-2"
          />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 600 }} />
      </SimulationMain>
      <SimulationResults>
        <div>Path length: {path.length}, Current step: {currentStep}, Final x: {path[path.length-1]?.toFixed(6)}</div>
      </SimulationResults>
    </SimulationPanel>
  );
};

export default Newton1D;
