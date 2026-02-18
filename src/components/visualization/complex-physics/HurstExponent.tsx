'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { usePlotlyTheme } from '@/lib/plotly-theme';
import { Slider } from '@/components/ui/slider';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

function generateBrownianTimeSeries(length: number, drift: number, volatility: number): number[] {
  const series: number[] = [100]; // starting price
  for (let i = 1; i < length; i++) {
    const dt = 1;
    const dW = (Math.random() + Math.random() + Math.random() + Math.random() +
                Math.random() + Math.random() - 3) * Math.sqrt(dt); // approx normal via CLT
    series.push(series[i - 1] * Math.exp((drift - 0.5 * volatility * volatility) * dt + volatility * dW));
  }
  return series;
}

function computeHurstExponent(timeSeries: number[], maxLag: number): number {
  const lags = Array.from({ length: maxLag - 1 }, (_, i) => i + 2);
  const logLags: number[] = [];
  const logTau: number[] = [];

  for (const lag of lags) {
    const diffs: number[] = [];
    for (let i = lag; i < timeSeries.length; i++) {
      diffs.push(timeSeries[i] - timeSeries[i - lag]);
    }
    if (diffs.length === 0) continue;

    const mean = diffs.reduce((s, v) => s + v, 0) / diffs.length;
    const variance = diffs.reduce((s, v) => s + (v - mean) ** 2, 0) / diffs.length;
    const std = Math.sqrt(variance);

    if (std > 0) {
      logLags.push(Math.log(lag));
      logTau.push(Math.log(std));
    }
  }

  // Linear regression in log-log space
  if (logLags.length < 2) return 0.5;
  const n = logLags.length;
  const sumX = logLags.reduce((s, v) => s + v, 0);
  const sumY = logTau.reduce((s, v) => s + v, 0);
  const sumXY = logLags.reduce((s, v, i) => s + v * logTau[i], 0);
  const sumX2 = logLags.reduce((s, v) => s + v * v, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  return slope;
}

function computeVarianceVsLag(timeSeries: number[]): { lags: number[]; vars: number[] } {
  const lags: number[] = [];
  const vars: number[] = [];

  for (let lag = 1; lag <= Math.min(30, Math.floor(timeSeries.length / 4)); lag++) {
    const diffs: number[] = [];
    for (let i = lag; i < timeSeries.length; i++) {
      diffs.push((timeSeries[i] - timeSeries[i - lag]) ** 2);
    }
    if (diffs.length > 0) {
      lags.push(lag);
      vars.push(diffs.reduce((s, v) => s + v, 0) / diffs.length);
    }
  }

  return { lags, vars };
}

export function HurstExponent() {
  const [seriesLength, setSeriesLength] = useState(500);
  const [drift, setDrift] = useState(0.0005);
  const [volatility, setVolatility] = useState(0.02);
  const [seed, setSeed] = useState(0);
  const { mergeLayout } = usePlotlyTheme();

  const { timeSeries, hurstValues, varData } = useMemo(() => {
    const ts = generateBrownianTimeSeries(seriesLength, drift, volatility);

    // Compute Hurst at various max lags
    const lagValues = Array.from(
      { length: 10 },
      (_, i) => Math.floor(5 + (i * (seriesLength - 5)) / 10)
    );
    const hurstValues = lagValues.map(lag => ({
      lag,
      H: computeHurstExponent(ts, lag),
    }));

    const varData = computeVarianceVsLag(ts);

    return { timeSeries: ts, hurstValues, varData };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seriesLength, drift, volatility, seed]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Series Length: {seriesLength}</label>
          <Slider
            min={100}
            max={2000}
            step={50}
            value={[seriesLength]}
            onValueChange={([v]) => setSeriesLength(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Drift: {drift.toFixed(4)}</label>
          <Slider
            min={-0.005}
            max={0.005}
            step={0.0001}
            value={[drift]}
            onValueChange={([v]) => setDrift(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Volatility: {volatility.toFixed(3)}</label>
          <Slider
            min={0.001}
            max={0.1}
            step={0.001}
            value={[volatility]}
            onValueChange={([v]) => setVolatility(v)}
            className="w-48"
          />
        </div>
        <button
          onClick={() => setSeed(s => s + 1)}
          className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4"
        >
          Re-generate
        </button>
      </div>

      {/* Time series and variance */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[{
            y: timeSeries,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#3b82f6', width: 1.5 },
          }]}
          layout={mergeLayout({
            title: { text: 'Simulated Price Series (GBM)', font: { size: 13 } },
            xaxis: { title: { text: 'Time' } },
            yaxis: { title: { text: 'Price' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 300 }}
        />
        <Plot
          data={[
            {
              x: varData.lags,
              y: varData.vars,
              type: 'scatter',
              mode: 'markers',
              marker: { color: '#10b981', size: 6 },
              name: 'Data',
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Variance vs Lag', font: { size: 13 } },
            xaxis: { title: { text: 'Lag (tau)' } },
            yaxis: { title: { text: 'var(tau)' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 300 }}
        />
      </div>

      {/* Hurst exponent */}
      <Plot
        data={[
          {
            x: hurstValues.map(h => h.lag),
            y: hurstValues.map(h => h.H),
            type: 'scatter',
            mode: 'markers',
            marker: { color: '#f59e0b', size: 8 },
            name: 'H',
          },
          {
            x: [hurstValues[0]?.lag ?? 0, hurstValues[hurstValues.length - 1]?.lag ?? 1],
            y: [0.5, 0.5],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ef4444', width: 1, dash: 'dash' },
            name: 'H = 0.5 (Brownian)',
          },
        ]}
        layout={mergeLayout({
          title: { text: 'Hurst Exponent at Various Max Lags', font: { size: 14 } },
          xaxis: { title: { text: 'Max Lag' } },
          yaxis: { title: { text: 'Hurst Exponent H' } },
          showlegend: true,
          margin: { t: 40, r: 20, b: 50, l: 60 },
        })}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: 300 }}
      />
    </div>
  );
}
