'use client';

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { mulberry32, gaussianPair, linspace } from './ml-utils';

/**
 * Fit a line y = a + b*x minimizing the given loss on (xs, ys).
 * Uses simple gradient descent.
 */
function fitLine(
  xs: number[],
  ys: number[],
  loss: 'mse' | 'mae' | 'huber',
  delta: number,
): { a: number; b: number } {
  let a = 0;
  let b = 0;
  const lr = 0.001;
  const n = xs.length;

  for (let iter = 0; iter < 600; iter++) {
    let ga = 0;
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const r = ys[i] - (a + b * xs[i]);
      let dr: number;
      if (loss === 'mse') {
        dr = -2 * r;
      } else if (loss === 'mae') {
        dr = r > 0 ? -1 : r < 0 ? 1 : 0;
      } else {
        // Huber
        dr = Math.abs(r) <= delta ? -2 * r : -2 * delta * Math.sign(r);
      }
      ga += dr / n;
      gb += (dr * xs[i]) / n;
    }
    a -= lr * ga;
    b -= lr * gb;
  }
  return { a, b };
}

export default function LossCurvesOutliers(): React.ReactElement {
  const [outlierStrength, setOutlierStrength] = useState(8);
  const [outlierCount, setOutlierCount] = useState(5);
  const [huberDelta, setHuberDelta] = useState(1.0);

  // Generate data with outliers
  const { xs, ys } = useMemo(() => {
    const rng = mulberry32(42);
    const n = 60;
    const xArr: number[] = [];
    const yArr: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = linspace(0, 10, n)[i];
      const [noise] = gaussianPair(rng);
      xArr.push(x);
      yArr.push(2 + 1.5 * x + noise * 0.8);
    }
    // Add outliers at the end
    for (let i = 0; i < outlierCount; i++) {
      const x = 2 + rng() * 6;
      xArr.push(x);
      yArr.push(2 + 1.5 * x + outlierStrength * (rng() > 0.5 ? 1 : -1));
    }
    return { xs: xArr, ys: yArr };
  }, [outlierStrength, outlierCount]);

  // Fit lines with each loss
  const fits = useMemo(() => {
    const mse = fitLine(xs, ys, 'mse', huberDelta);
    const mae = fitLine(xs, ys, 'mae', huberDelta);
    const huber = fitLine(xs, ys, 'huber', huberDelta);
    return { mse, mae, huber };
  }, [xs, ys, huberDelta]);

  // Build fit line traces
  const xLine = linspace(0, 10, 100);
  const mseLine = xLine.map((x) => fits.mse.a + fits.mse.b * x);
  const maeLine = xLine.map((x) => fits.mae.a + fits.mae.b * x);
  const huberLine = xLine.map((x) => fits.huber.a + fits.huber.b * x);

  // Separate regular and outlier points for coloring
  const nRegular = 60;
  const regularColors = xs.map((_, i) =>
    i < nRegular ? '#3b82f6' : '#ef4444',
  );

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Loss Curves Under Outliers
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        MSE (green) is dragged toward red outlier points. MAE (amber) and Huber (pink) stay anchored to the majority.
      </p>

      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <label className="text-sm text-[var(--text-muted)]">
            Outlier strength: {outlierStrength.toFixed(1)}
          </label>
          <Slider
            min={1}
            max={15}
            step={0.5}
            value={[outlierStrength]}
            onValueChange={([v]) => setOutlierStrength(v)}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">
            Outlier count: {outlierCount}
          </label>
          <Slider
            min={1}
            max={15}
            step={1}
            value={[outlierCount]}
            onValueChange={([v]) => setOutlierCount(v)}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">
            Huber delta: {huberDelta.toFixed(2)}
          </label>
          <Slider
            min={0.1}
            max={5}
            step={0.1}
            value={[huberDelta]}
            onValueChange={([v]) => setHuberDelta(v)}
          />
        </div>
      </div>

      <CanvasChart
        data={[
          {
            x: xs,
            y: ys,
            type: 'scatter',
            mode: 'markers',
            marker: { color: regularColors, size: 7, opacity: 0.7 },
            name: 'Data',
            showlegend: false,
          },
          {
            x: xLine,
            y: mseLine,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#10b981', width: 2.5 },
            name: 'MSE fit',
          },
          {
            x: xLine,
            y: maeLine,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#f59e0b', width: 2.5 },
            name: 'MAE fit',
          },
          {
            x: xLine,
            y: huberLine,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ec4899', width: 2.5, dash: 'dash' },
            name: 'Huber fit',
          },
        ]}
        layout={{
          xaxis: { title: { text: 'x' } },
          yaxis: { title: { text: 'y' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 420 }}
      />

      <div className="mt-3 text-xs text-[var(--text-muted)]">
        Blue dots are clean data; red dots are outliers. Increase outlier strength to see MSE deviate.
      </div>
    </div>
  );
}
