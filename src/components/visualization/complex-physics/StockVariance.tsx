'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const STOCK_PRESETS: Record<string, { drift: number; vol: number }> = {
  GOOGL: { drift: 0.0006, vol: 0.018 },
  AAPL: { drift: 0.0008, vol: 0.021 },
  TSLA: { drift: 0.001, vol: 0.035 },
};

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-10);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function generateStockSeries(n: number, drift: number, vol: number, rand: () => number): number[] {
  const series = [100];
  for (let i = 1; i < n; i++) {
    const shock = gaussian(rand);
    const prev = series[i - 1];
    const next = prev * Math.exp((drift - 0.5 * vol * vol) + vol * shock);
    series.push(next);
  }
  return series;
}

function varianceVsLag(logSeries: number[], maxLag: number): { lags: number[]; variances: number[] } {
  const lags: number[] = [];
  const variances: number[] = [];
  for (let lag = 1; lag <= maxLag; lag++) {
    const vals: number[] = [];
    for (let i = lag; i < logSeries.length; i++) {
      vals.push((logSeries[i] - logSeries[i - lag]) ** 2);
    }
    if (vals.length > 0) {
      lags.push(lag);
      variances.push(vals.reduce((s, v) => s + v, 0) / vals.length);
    }
  }
  return { lags, variances };
}

export function StockVariance() {
  const [ticker, setTicker] = useState<'GOOGL' | 'AAPL' | 'TSLA'>('GOOGL');
  const [length, setLength] = useState(600);
  const [noiseScale, setNoiseScale] = useState(1);
  const [rerun, setRerun] = useState(0);

  const { prices, varData } = useMemo(() => {
    const rand = mulberry32((length * 31 + Math.floor(noiseScale * 100) * 37 + rerun * 101 + ticker.charCodeAt(0) * 97) >>> 0);
    const preset = STOCK_PRESETS[ticker];
    const prices = generateStockSeries(length, preset.drift, preset.vol * noiseScale, rand);
    const logPrices = prices.map(v => Math.log(v));
    const varData = varianceVsLag(logPrices, Math.min(60, Math.floor(length / 5)));
    return { prices, varData };
  }, [ticker, length, noiseScale, rerun]);

  const darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#9ca3af' },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    xaxis: { gridcolor: '#1e1e2e' },
    yaxis: { gridcolor: '#1e1e2e' },
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Ticker</label>
          <select value={ticker} onChange={e => setTicker(e.target.value as 'GOOGL' | 'AAPL' | 'TSLA')} className="bg-[#0f0f19] border border-gray-700 rounded px-3 py-2 text-sm text-gray-200">
            <option value="GOOGL">GOOGL</option>
            <option value="AAPL">AAPL</option>
            <option value="TSLA">TSLA</option>
          </select>
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Series Length: {length}</label>
          <input type="range" min={200} max={1500} step={50} value={length} onChange={e => setLength(Number(e.target.value))} className="w-48" />
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Noise Scale: {noiseScale.toFixed(2)}</label>
          <input type="range" min={0.4} max={2.5} step={0.05} value={noiseScale} onChange={e => setNoiseScale(Number(e.target.value))} className="w-48" />
        </div>
        <button onClick={() => setRerun(v => v + 1)} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm mt-4">
          Re-simulate
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[{ y: prices, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 1.5 } }]}
          layout={{
            ...darkLayout,
            title: { text: `${ticker} Simulated Closing Price`, font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'Time index' } },
            yaxis: { ...darkLayout.yaxis, title: { text: 'Price' } },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
        <Plot
          data={[{ x: varData.lags, y: varData.variances, type: 'scatter', mode: 'lines+markers', line: { color: '#34d399', width: 2 }, marker: { size: 5 } }]}
          layout={{
            ...darkLayout,
            title: { text: 'var(tau) of log-price increments', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'tau' } },
            yaxis: { ...darkLayout.yaxis, title: { text: 'var(tau)' } },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 320 }}
        />
      </div>
    </div>
  );
}
