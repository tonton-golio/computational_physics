"use client";

import { useEffect, useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { runLotkaVolterraWorker, type LotkaVolterraResult } from '@/features/simulation/simulation-worker.client';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function solveLotkaVolterra(alpha: number, beta: number, gamma: number, delta: number, x0: number, y0: number, dt: number, steps: number): { t: number[], x: number[], y: number[] } {
  const t: number[] = [];
  const x: number[] = [];
  const y: number[] = [];

  let currentX = x0;
  let currentY = y0;
  let currentT = 0;

  for (let i = 0; i < steps; i++) {
    t.push(currentT);
    x.push(currentX);
    y.push(currentY);

    const dx = (alpha * currentX - beta * currentX * currentY) * dt;
    const dy = (-gamma * currentY + delta * currentX * currentY) * dt;

    currentX += dx;
    currentY += dy;
    currentT += dt;
  }

  return { t, x, y };
}

function computeBifurcation(beta: number, gamma: number, delta: number, alphaRange: number[], x0: number, y0: number): { alpha: number[], minX: number[], maxX: number[], minY: number[], maxY: number[] } {
  const alpha: number[] = [];
  const minX: number[] = [];
  const maxX: number[] = [];
  const minY: number[] = [];
  const maxY: number[] = [];

  alphaRange.forEach(a => {
    const solution = solveLotkaVolterra(a, beta, gamma, delta, x0, y0, 0.01, 10000);
    const xs = solution.x.slice(1000); // skip transient
    const ys = solution.y.slice(1000);
    alpha.push(a);
    minX.push(Math.min(...xs));
    maxX.push(Math.max(...xs));
    minY.push(Math.min(...ys));
    maxY.push(Math.max(...ys));
  });

  return { alpha, minX, maxX, minY, maxY };
}

const bifurcationCache = new Map<string, { alpha: number[]; minX: number[]; maxX: number[]; minY: number[]; maxY: number[] }>();
const MAX_BIFURCATION_CACHE_ENTRIES = 16;

function getBifurcationCacheKey(beta: number, gamma: number, delta: number, x0: number, y0: number): string {
  return [beta, gamma, delta, x0, y0].map((value) => value.toFixed(6)).join('|');
}

function getCachedBifurcation(
  beta: number,
  gamma: number,
  delta: number,
  x0: number,
  y0: number,
  alphaRange: number[]
): { alpha: number[]; minX: number[]; maxX: number[]; minY: number[]; maxY: number[] } {
  const key = getBifurcationCacheKey(beta, gamma, delta, x0, y0);
  const cached = bifurcationCache.get(key);
  if (cached) {
    bifurcationCache.delete(key);
    bifurcationCache.set(key, cached);
    return cached;
  }

  const result = computeBifurcation(beta, gamma, delta, alphaRange, x0, y0);
  bifurcationCache.set(key, result);
  if (bifurcationCache.size > MAX_BIFURCATION_CACHE_ENTRIES) {
    const oldestKey = bifurcationCache.keys().next().value;
    if (oldestKey) {
      bifurcationCache.delete(oldestKey);
    }
  }
  return result;
}

export default function LotkaVolterraSim({}: SimulationComponentProps) {
  const [alpha, setAlpha] = useState([1.0]);
  const [beta, setBeta] = useState([0.1]);
  const [gamma, setGamma] = useState([1.5]);
  const [delta, setDelta] = useState([0.075]);
  const [x0, setX0] = useState([10]);
  const [y0, setY0] = useState([5]);
  const [workerSolution, setWorkerSolution] = useState<LotkaVolterraResult | null>(null);
  const alphaValue = alpha[0];
  const betaValue = beta[0];
  const gammaValue = gamma[0];
  const deltaValue = delta[0];
  const x0Value = x0[0];
  const y0Value = y0[0];

  useEffect(() => {
    let active = true;
    void runLotkaVolterraWorker({
      alpha: alphaValue,
      beta: betaValue,
      gamma: gammaValue,
      delta: deltaValue,
      x0: x0Value,
      y0: y0Value,
      dt: 0.01,
      steps: 2000,
    })
      .then((result) => {
        if (!active) return;
        setWorkerSolution(result);
      })
      .catch(() => {
        if (!active) return;
        setWorkerSolution(null);
      });

    return () => {
      active = false;
    };
  }, [alphaValue, betaValue, gammaValue, deltaValue, x0Value, y0Value]);

  const fallbackSolution = useMemo(
    () => solveLotkaVolterra(alphaValue, betaValue, gammaValue, deltaValue, x0Value, y0Value, 0.01, 2000),
    [alphaValue, betaValue, gammaValue, deltaValue, x0Value, y0Value]
  );
  const baseSolution = workerSolution ?? fallbackSolution;

  const timeSeriesData = useMemo(() => {
    return [
      {
        x: baseSolution.t,
        y: baseSolution.x,
        mode: 'lines',
        name: 'Prey',
        line: { color: 'blue' },
      },
      {
        x: baseSolution.t,
        y: baseSolution.y,
        mode: 'lines',
        name: 'Predator',
        line: { color: 'red' },
      },
    ];
  }, [baseSolution]);

  const phasePortraitData = useMemo(() => {
    return [
      {
        x: baseSolution.x,
        y: baseSolution.y,
        mode: 'lines',
        type: 'scatter',
        line: { color: 'green' },
      },
    ];
  }, [baseSolution]);

  const bifurcationData = useMemo(() => {
    const alphaRange = [];
    for (let a = 0.1; a <= 3; a += 0.1) {
      alphaRange.push(a);
    }
    const bif = getCachedBifurcation(betaValue, gammaValue, deltaValue, x0Value, y0Value, alphaRange);
    return [
      {
        x: bif.alpha,
        y: bif.minX,
        mode: 'lines',
        name: 'Min Prey',
        line: { color: 'lightblue' },
        showlegend: false,
      },
      {
        x: bif.alpha,
        y: bif.maxX,
        mode: 'lines',
        name: 'Max Prey',
        line: { color: 'blue' },
        fill: 'tonexty',
        fillcolor: 'rgba(0,0,255,0.1)',
      },
      {
        x: bif.alpha,
        y: bif.minY,
        mode: 'lines',
        name: 'Min Predator',
        line: { color: 'pink' },
        showlegend: false,
      },
      {
        x: bif.alpha,
        y: bif.maxY,
        mode: 'lines',
        name: 'Max Predator',
        line: { color: 'red' },
        fill: 'tonexty',
        fillcolor: 'rgba(255,0,0,0.1)',
      },
    ];
  }, [betaValue, gammaValue, deltaValue, x0Value, y0Value]);

  return (
    <SimulationPanel title="Lotka-Volterra Predator-Prey Model">
      <SimulationConfig>
        <div>
          <SimulationLabel>Alpha (prey growth): {alpha[0].toFixed(2)}</SimulationLabel>
          <Slider value={alpha} onValueChange={setAlpha} min={0} max={3} step={0.01} />
        </div>
        <div>
          <SimulationLabel>Beta (predation): {beta[0].toFixed(3)}</SimulationLabel>
          <Slider value={beta} onValueChange={setBeta} min={0} max={0.5} step={0.001} />
        </div>
        <div>
          <SimulationLabel>Gamma (predator death): {gamma[0].toFixed(2)}</SimulationLabel>
          <Slider value={gamma} onValueChange={setGamma} min={0} max={3} step={0.01} />
        </div>
        <div>
          <SimulationLabel>Delta (predator efficiency): {delta[0].toFixed(3)}</SimulationLabel>
          <Slider value={delta} onValueChange={setDelta} min={0} max={0.2} step={0.001} />
        </div>
        <div>
          <SimulationLabel>Initial Prey: {x0[0].toFixed(1)}</SimulationLabel>
          <Slider value={x0} onValueChange={setX0} min={1} max={50} step={0.1} />
        </div>
        <div>
          <SimulationLabel>Initial Predator: {y0[0].toFixed(1)}</SimulationLabel>
          <Slider value={y0} onValueChange={setY0} min={1} max={50} step={0.1} />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div>
          <h3 className="text-lg font-semibold mb-4">Time Series</h3>
          {timeSeriesData && (
            <CanvasChart
              data={timeSeriesData as any}
              layout={{
                width: 400,
                height: 300,
                xaxis: { title: { text: 'Time' } },
                yaxis: { title: { text: 'Population' } },
                margin: { t: 20, b: 40, l: 60, r: 20 },
              }}
            />
          )}
        </div>
        <div>
          <h3 className="text-lg font-semibold mb-4">Phase Portrait</h3>
          {phasePortraitData && (
            <CanvasChart
              data={phasePortraitData as any}
              layout={{
                width: 400,
                height: 300,
                xaxis: { title: { text: 'Prey' } },
                yaxis: { title: { text: 'Predator' } },
                margin: { t: 20, b: 40, l: 60, r: 20 },
              }}
            />
          )}
        </div>
        <div>
          <h3 className="text-lg font-semibold mb-4">Bifurcation Diagram (vs Alpha)</h3>
          {bifurcationData && (
            <CanvasChart
              data={bifurcationData as any}
              layout={{
                width: 400,
                height: 300,
                xaxis: { title: { text: 'Alpha' } },
                yaxis: { title: { text: 'Population' } },
                margin: { t: 20, b: 40, l: 60, r: 20 },
              }}
            />
          )}
        </div>
      </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
