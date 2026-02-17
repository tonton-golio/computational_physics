'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const WIDTH = 400;
const HEIGHT = 400;
const MAX_ITER = 100;

function mandelbrot(c: { re: number; im: number }): number {
  const z = { re: 0, im: 0 };
  let n = 0;
  while (z.re * z.re + z.im * z.im <= 4 && n < MAX_ITER) {
    const temp = z.re * z.re - z.im * z.im + c.re;
    z.im = 2 * z.re * z.im + c.im;
    z.re = temp;
    n++;
  }
  return n;
}

export function MandelbrotFractal() {
  const [xRange, setXRange] = useState([-2, 1]);
  const [yRange, setYRange] = useState([-1.5, 1.5]);

  const data = useMemo(() => {
    const xMin = xRange[0];
    const xMax = xRange[1];
    const yMin = yRange[0];
    const yMax = yRange[1];

    const xStep = (xMax - xMin) / WIDTH;
    const yStep = (yMax - yMin) / HEIGHT;

    const z: number[][] = [];
    for (let i = 0; i < HEIGHT; i++) {
      const row: number[] = [];
      for (let j = 0; j < WIDTH; j++) {
        const c = {
          re: xMin + j * xStep,
          im: yMin + i * yStep,
        };
        row.push(mandelbrot(c));
      }
      z.push(row);
    }

    return [
      {
        z,
        type: 'heatmap' as const,
        colorscale: 'Viridis',
        showscale: false,
      },
    ];
  }, [xRange, yRange]);

  const handleRelayout = (event: Record<string, unknown>) => {
    if (event['xaxis.range[0]'] && event['xaxis.range[1]']) {
      setXRange([event['xaxis.range[0]'] as number, event['xaxis.range[1]'] as number]);
    }
    if (event['yaxis.range[0]'] && event['yaxis.range[1]']) {
      setYRange([event['yaxis.range[0]'] as number, event['yaxis.range[1]'] as number]);
    }
  };

  return (
    <div>
      {data && (
        <Plot
          data={data}
          layout={{
            width: 600,
            height: 600,
            xaxis: { autorange: false, range: xRange },
            yaxis: { autorange: false, range: yRange },
            margin: { t: 0, b: 0, l: 0, r: 0 },
          }}
          config={{ displayModeBar: true }}
          onRelayout={handleRelayout}
        />
      )}
    </div>
  );
}