'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

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

export function LotkaVolterraSim() {
  const [alpha, setAlpha] = useState([1.0]);
  const [beta, setBeta] = useState([0.1]);
  const [gamma, setGamma] = useState([1.5]);
  const [delta, setDelta] = useState([0.075]);
  const [x0, setX0] = useState([10]);
  const [y0, setY0] = useState([5]);

  const [timeSeriesData, setTimeSeriesData] = useState<any>(null);
  const [phasePortraitData, setPhasePortraitData] = useState<any>(null);
  const [bifurcationData, setBifurcationData] = useState<any>(null);

  useEffect(() => {
    const solution = solveLotkaVolterra(alpha[0], beta[0], gamma[0], delta[0], x0[0], y0[0], 0.01, 2000);

    setTimeSeriesData([
      {
        x: solution.t,
        y: solution.x,
        mode: 'lines',
        name: 'Prey',
        line: { color: 'blue' },
      },
      {
        x: solution.t,
        y: solution.y,
        mode: 'lines',
        name: 'Predator',
        line: { color: 'red' },
      },
    ]);

    setPhasePortraitData([
      {
        x: solution.x,
        y: solution.y,
        mode: 'lines',
        type: 'scatter',
        line: { color: 'green' },
      },
    ]);

    const alphaRange = [];
    for (let a = 0.1; a <= 3; a += 0.1) {
      alphaRange.push(a);
    }
    const bif = computeBifurcation(beta[0], gamma[0], delta[0], alphaRange, x0[0], y0[0]);
    setBifurcationData([
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
    ]);
  }, [beta, gamma, delta, x0, y0]);

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Lotka-Volterra Predator-Prey Model</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-4">Time Series</h3>
              {timeSeriesData && (
                <Plot
                  data={timeSeriesData}
                  layout={{
                    width: 400,
                    height: 300,
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Population' },
                    margin: { t: 20, b: 40, l: 60, r: 20 },
                  }}
                  config={{ displayModeBar: false }}
                />
              )}
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Phase Portrait</h3>
              {phasePortraitData && (
                <Plot
                  data={phasePortraitData}
                  layout={{
                    width: 400,
                    height: 300,
                    xaxis: { title: 'Prey' },
                    yaxis: { title: 'Predator' },
                    margin: { t: 20, b: 40, l: 60, r: 20 },
                  }}
                  config={{ displayModeBar: false }}
                />
              )}
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Bifurcation Diagram (vs Alpha)</h3>
              {bifurcationData && (
                <Plot
                  data={bifurcationData}
                  layout={{
                    width: 400,
                    height: 300,
                    xaxis: { title: 'Alpha' },
                    yaxis: { title: 'Population' },
                    margin: { t: 20, b: 40, l: 60, r: 20 },
                  }}
                  config={{ displayModeBar: false }}
                />
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Parameters</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="text-sm font-medium">Alpha (prey growth): {alpha[0].toFixed(2)}</label>
            <Slider value={alpha} onValueChange={setAlpha} min={0} max={3} step={0.01} />
          </div>
          <div>
            <label className="text-sm font-medium">Beta (predation): {beta[0].toFixed(3)}</label>
            <Slider value={beta} onValueChange={setBeta} min={0} max={0.5} step={0.001} />
          </div>
          <div>
            <label className="text-sm font-medium">Gamma (predator death): {gamma[0].toFixed(2)}</label>
            <Slider value={gamma} onValueChange={setGamma} min={0} max={3} step={0.01} />
          </div>
          <div>
            <label className="text-sm font-medium">Delta (predator efficiency): {delta[0].toFixed(3)}</label>
            <Slider value={delta} onValueChange={setDelta} min={0} max={0.2} step={0.001} />
          </div>
          <div>
            <label className="text-sm font-medium">Initial Prey: {x0[0].toFixed(1)}</label>
            <Slider value={x0} onValueChange={setX0} min={1} max={50} step={0.1} />
          </div>
          <div>
            <label className="text-sm font-medium">Initial Predator: {y0[0].toFixed(1)}</label>
            <Slider value={y0} onValueChange={setY0} min={1} max={50} step={0.1} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}