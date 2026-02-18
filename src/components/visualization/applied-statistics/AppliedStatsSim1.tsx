'use client';

import React, { useState, useEffect } from 'react';
import * as math from 'mathjs';
import Plotly from 'react-plotly.js';
import { usePlotlyTheme } from '@/lib/plotly-theme';
import { Slider } from '@/components/ui/slider';

interface SimulationProps {
  id: string;
}

export default function AppliedStatsSim1({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [noise, setNoise] = useState(1);
  const [sampleSize, setSampleSize] = useState(20);
  const { mergeLayout } = usePlotlyTheme();

  const [data, setData] = useState<{
    x: number[],
    y: number[],
    yHat: number[],
    residuals: number[],
    ciUpper: number[],
    ciLower: number[],
    rSquared: number,
    chiSquare: number,
    beta0: number,
    beta1: number
  } | null>(null);

  useEffect(() => {
    const trueBeta0 = 2;
    const trueBeta1 = 1.5;
    const x = Array.from({length: sampleSize}, () => math.random(0, 10));
    const y = x.map(xi => trueBeta0 + trueBeta1 * xi + (Math.random() - 0.5) * noise * 2);
    const meanX = math.number(math.mean(x));
    const meanY = math.number(math.mean(y));
    const covXY = math.number(math.mean(x.map((xi, i) => (xi - meanX) * (y[i] - meanY))));
    const varX = math.number(math.variance(x));
    const beta1 = Number(covXY) / Number(varX);
    const beta0 = meanY - beta1 * meanX;
    const yHat = x.map(xi => beta0 + beta1 * xi);
    const residuals = y.map((yi, i) => yi - yHat[i]);
    const ssRes = math.number(math.sum(residuals.map(e => e * e)));
    const ssTot = math.number(math.sum(y.map(yi => (yi - meanY) ** 2)));
    const rSquared = 1 - ssRes / ssTot;
    const sigmaHat = Math.sqrt(ssRes / (sampleSize - 2));
    const chiSquare = ssRes / (sigmaHat ** 2); // assuming constant variance
    const sumSqX = math.number(math.sum(x.map(xi => (xi - meanX) ** 2)));
    const t = 2; // approx for large n
    const ciUpper = x.map((xi, i) => {
      const se = sigmaHat * Math.sqrt(1 / sampleSize + (xi - meanX) ** 2 / sumSqX);
      return yHat[i] + t * se;
    });
    const ciLower = x.map((xi, i) => {
      const se = sigmaHat * Math.sqrt(1 / sampleSize + (xi - meanX) ** 2 / sumSqX);
      return yHat[i] - t * se;
    });

    setData({ x, y, yHat, residuals, ciUpper, ciLower, rSquared, chiSquare, beta0, beta1 });
  }, [noise, sampleSize]);

  if (!data) return <div>Loading...</div>;

  const residualTraces = data.x.map((xi, i) => ({
    x: [xi, xi],
    y: [data.y[i], data.yHat[i]],
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { color: 'red', width: 1 },
    showlegend: i === 0,
    name: 'Residuals'
  }));

  const plotData = [
    {
      x: data.x,
      y: data.y,
      type: 'scatter' as const,
      mode: 'markers' as const,
      name: 'Data Points',
      marker: { color: 'blue' }
    },
    {
      x: data.x,
      y: data.yHat,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Fitted Line',
      line: { color: 'green' }
    },
    {
      x: data.x,
      y: data.ciUpper,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: 'gray' },
      showlegend: false
    },
    {
      x: data.x,
      y: data.ciLower,
      type: 'scatter' as const,
      mode: 'lines' as const,
      fill: 'tonexty',
      fillcolor: 'rgba(128,128,128,0.2)',
      line: { color: 'gray' },
      name: '95% CI'
    },
    ...residualTraces
  ];

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Interactive Linear Regression Simulation</h3>
      <div className="mb-4">
        <p className="text-sm text-[var(--text-muted)]">
          Generate noisy data points around a true linear relationship (y = 2 + 1.5x + e), fit a line using least squares,
          and visualize the fitted line, residuals, and confidence intervals.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-[var(--text-strong)]">Noise Level (sigma): {noise.toFixed(1)}</label>
          <Slider
            min={0.1}
            max={5}
            step={0.1}
            value={[noise]}
            onValueChange={([v]) => setNoise(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Sample Size: {sampleSize}</label>
          <Slider
            min={10}
            max={100}
            step={5}
            value={[sampleSize]}
            onValueChange={([v]) => setSampleSize(v)}
            className="w-full"
          />
        </div>
      </div>
      <div className="mb-4 text-[var(--text-muted)]">
        <p>Fitted Line: y = {data.beta0.toFixed(3)} + {data.beta1.toFixed(3)}x</p>
        <p>R-squared: {data.rSquared.toFixed(3)}</p>
        <p>Chi-square: {data.chiSquare.toFixed(3)}</p>
      </div>
      <Plotly
        data={plotData as any}
        layout={mergeLayout({
          title: { text: 'Linear Regression: Data, Fit, Residuals, and 95% CI' },
          xaxis: { title: { text: 'x' } },
          yaxis: { title: { text: 'y' } },
          height: 500,
        })}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}
