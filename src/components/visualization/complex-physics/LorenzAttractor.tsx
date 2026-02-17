'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function computeLorenz(sigma: number, rho: number, beta: number): { x: number[]; y: number[]; z: number[] } {
  const dt = 0.01;
  const steps = 10000;
  let x = 1;
  let y = 1;
  let z = 1;

  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];

  for (let i = 0; i < steps; i++) {
    const dx = sigma * (y - x);
    const dy = x * (rho - z) - y;
    const dz = x * y - beta * z;

    x += dx * dt;
    y += dy * dt;
    z += dz * dt;

    xs.push(x);
    ys.push(y);
    zs.push(z);
  }

  return { x: xs, y: ys, z: zs };
}

export function LorenzAttractor() {
  const [sigma, setSigma] = useState([10]);
  const [rho, setRho] = useState([28]);
  const [beta, setBeta] = useState([8/3]);

  const data = useMemo(() => {
    const trajectory = computeLorenz(sigma[0], rho[0], beta[0]);
    return [
      {
        x: trajectory.x,
        y: trajectory.y,
        z: trajectory.z,
        mode: 'lines',
        type: 'scatter3d',
        line: { color: 'blue', width: 1 },
      },
    ];
  }, [sigma, rho, beta]);

  return (
    <div className="flex flex-col lg:flex-row gap-4">
      <div className="flex-1">
        {data && (
          <Plot
            data={data}
            layout={{
              width: 600,
              height: 600,
              scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Z' },
              },
              margin: { t: 0, b: 0, l: 0, r: 0 },
            }}
            config={{ displayModeBar: true }}
          />
        )}
      </div>
      <Card className="w-full lg:w-80">
        <CardHeader>
          <CardTitle>Parameters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium">Sigma: {sigma[0].toFixed(1)}</label>
            <Slider value={sigma} onValueChange={setSigma} min={0} max={20} step={0.1} />
          </div>
          <div>
            <label className="text-sm font-medium">Rho: {rho[0].toFixed(1)}</label>
            <Slider value={rho} onValueChange={setRho} min={0} max={50} step={0.1} />
          </div>
          <div>
            <label className="text-sm font-medium">Beta: {beta[0].toFixed(2)}</label>
            <Slider value={beta} onValueChange={setBeta} min={0} max={5} step={0.01} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}