'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface BakSneppenResult {
  chains: number[][];
  meanValues: number[];
  idxArr: number[];
  skipInit: number;
  avalancheTSpans: number[];
  avalancheXSpans: number[];
}

function runBakSneppen(size: number, nsteps: number): BakSneppenResult {
  const chain = Array.from({ length: size }, () => Math.random());
  const chains: number[][] = [];
  const idxArr: number[] = [];

  for (let n = 0; n < nsteps; n++) {
    let minIdx = 0;
    let minVal = chain[0];
    for (let i = 1; i < size; i++) {
      if (chain[i] < minVal) {
        minVal = chain[i];
        minIdx = i;
      }
    }
    idxArr.push(minIdx);

    // Replace min and its neighbors with new random values
    for (const offset of [-1, 0, 1]) {
      const idx = ((minIdx + offset) % size + size) % size;
      chain[idx] = Math.random();
    }
    chains.push([...chain]);
  }

  // Calculate mean values over time
  const meanValues = chains.map(c => c.reduce((s, v) => s + v, 0) / c.length);

  // Find skip-init point (when mean stabilizes)
  const patience = Math.min(500, Math.floor(nsteps / 4));
  let skipInit = patience;
  for (let i = patience; i < meanValues.length; i++) {
    const windowMean = meanValues.slice(i - patience, i).reduce((s, v) => s + v, 0) / patience;
    if (Math.abs(meanValues[i] - windowMean) < 0.001) {
      skipInit = i;
      break;
    }
  }

  // Compute avalanches from steady state
  const steadyIdx = idxArr.slice(skipInit);
  const avalancheTSpans: number[] = [];
  const avalancheXSpans: number[] = [];
  let counter = 0;
  const indices: number[] = [];

  for (let i = 1; i < steadyIdx.length; i++) {
    const isAvalanche = Math.abs(steadyIdx[i] - steadyIdx[i - 1]) < 2;
    if (isAvalanche) {
      counter++;
      indices.push(steadyIdx[i]);
    } else if (counter > 0) {
      avalancheTSpans.push(counter);
      avalancheXSpans.push(Math.max(...indices) - Math.min(...indices) + 1);
      counter = 0;
      indices.length = 0;
    }
  }

  return { chains, meanValues, idxArr, skipInit, avalancheTSpans, avalancheXSpans };
}

export function BakSneppen() {
  const [size, setSize] = useState(200);
  const [nsteps, setNsteps] = useState(8000);
  const [seed, setSeed] = useState(0);

  const result = useMemo(() => {
    return runBakSneppen(size, nsteps);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, nsteps, seed]);

  const darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#9ca3af' },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: '#1e1e2e' },
    yaxis: { gridcolor: '#1e1e2e' },
  };

  // Subsample the chains for the imshow display (take every Nth row)
  const subsampleRate = Math.max(1, Math.floor(result.chains.length / 300));
  const subsampledChains = result.chains.filter((_, i) => i % subsampleRate === 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-gray-400 block mb-1">Chain Size: {size}</label>
          <input
            type="range"
            min={50}
            max={500}
            step={10}
            value={size}
            onChange={e => setSize(Number(e.target.value))}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-gray-400 block mb-1">Steps: {nsteps}</label>
          <input
            type="range"
            min={1000}
            max={30000}
            step={500}
            value={nsteps}
            onChange={e => setNsteps(Number(e.target.value))}
            className="w-48"
          />
        </div>
        <button
          onClick={() => setSeed(s => s + 1)}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm mt-4"
        >
          Re-run
        </button>
      </div>

      {/* Chain evolution heatmap */}
      <Plot
        data={[{
          z: subsampledChains,
          type: 'heatmap',
          colorscale: 'RdYlBu',
          zmin: 0,
          zmax: 1,
          showscale: true,
          colorbar: { tickfont: { color: '#9ca3af' } },
        }]}
        layout={{
          ...darkLayout,
          title: { text: 'Bak-Sneppen Evolution (chain values over time)', font: { size: 13, color: '#9ca3af' } },
          xaxis: { ...darkLayout.xaxis, title: { text: 'Site index' } },
          yaxis: { ...darkLayout.yaxis, title: { text: 'Timestep (subsampled)' } },
          margin: { t: 40, r: 80, b: 50, l: 60 },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: 350 }}
      />

      {/* Mean value and argmin plots */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[
            {
              y: result.meanValues,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#3b82f6', width: 1 },
              name: 'Mean value',
            },
            {
              x: [result.skipInit, result.skipInit],
              y: [Math.min(...result.meanValues), Math.max(...result.meanValues)],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#ef4444', width: 2, dash: 'dash' },
              name: 'Steady state',
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: 'Average Value Over Time', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'Timestep' } },
            yaxis: { ...darkLayout.yaxis, title: { text: 'Mean' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 280 }}
        />
        <Plot
          data={[
            {
              y: result.idxArr,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#10b981', width: 1 },
              name: 'Argmin index',
            },
            {
              x: [result.skipInit, result.skipInit],
              y: [0, size],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#ef4444', width: 2, dash: 'dash' },
              name: 'Steady state',
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: 'Min-fitness Index Over Time', font: { size: 13, color: '#9ca3af' } },
            xaxis: { ...darkLayout.xaxis, title: { text: 'Timestep' } },
            yaxis: { ...darkLayout.yaxis, title: { text: 'Argmin' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 280 }}
        />
      </div>

      {/* Avalanche distributions */}
      {result.avalancheTSpans.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Plot
            data={[{
              x: result.avalancheTSpans,
              type: 'histogram',
              marker: { color: '#8b5cf6' },
              nbinsx: 20,
            }]}
            layout={{
              ...darkLayout,
              title: { text: 'Avalanche Duration Distribution', font: { size: 13, color: '#9ca3af' } },
              xaxis: { ...darkLayout.xaxis, title: { text: 'Duration (timesteps)' }, type: 'log' },
              yaxis: { ...darkLayout.yaxis, title: { text: 'Frequency' }, type: 'log' },
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%', height: 280 }}
          />
          <Plot
            data={[{
              x: result.avalancheXSpans,
              type: 'histogram',
              marker: { color: '#ec4899' },
              nbinsx: 20,
            }]}
            layout={{
              ...darkLayout,
              title: { text: 'Avalanche Spatial Span Distribution', font: { size: 13, color: '#9ca3af' } },
              xaxis: { ...darkLayout.xaxis, title: { text: 'Spatial span' }, type: 'log' },
              yaxis: { ...darkLayout.yaxis, title: { text: 'Frequency' }, type: 'log' },
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%', height: 280 }}
          />
        </div>
      )}
    </div>
  );
}
