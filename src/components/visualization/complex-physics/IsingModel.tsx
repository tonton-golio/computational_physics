'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { usePlotlyTheme } from '@/lib/plotly-theme';
import { Slider } from '@/components/ui/slider';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface IsingResult {
  grid: number[][];
  energyHistory: number[];
  magnetizationHistory: number[];
  snapshots: { step: number; grid: number[][] }[];
}

function runIsing(size: number, nsteps: number, beta: number, nsnapshots: number): IsingResult {
  // Initialize random grid
  const grid: number[][] = [];
  for (let i = 0; i < size; i++) {
    const row: number[] = [];
    for (let j = 0; j < size; j++) {
      row.push(Math.random() > 0.5 ? 1 : -1);
    }
    grid.push(row);
  }

  // Calculate initial energy
  let E = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const sumNeighbors =
        grid[i][(j + 1) % size] +
        grid[i][(j - 1 + size) % size] +
        grid[(i + 1) % size][j] +
        grid[(i - 1 + size) % size][j];
      E += -grid[i][j] * sumNeighbors / 2;
    }
  }

  const energyHistory: number[] = [E];
  const magnetizationHistory: number[] = [grid.flat().reduce((s, v) => s + v, 0)];
  const snapshots: { step: number; grid: number[][] }[] = [];
  const snapshotSteps = Array.from({ length: nsnapshots }, (_, i) => Math.floor(i * nsteps / nsnapshots));

  for (let step = 0; step < nsteps; step++) {
    const i = Math.floor(Math.random() * size);
    const j = Math.floor(Math.random() * size);

    const sumNeighbors =
      grid[i][(j + 1) % size] +
      grid[i][(j - 1 + size) % size] +
      grid[(i + 1) % size][j] +
      grid[(i - 1 + size) % size][j];

    const dE = 2 * grid[i][j] * sumNeighbors;

    if (Math.random() < Math.exp(-beta * dE)) {
      grid[i][j] *= -1;
      E += dE;
    }

    energyHistory.push(E);
    magnetizationHistory.push(grid.flat().reduce((s, v) => s + v, 0));

    if (snapshotSteps.includes(step)) {
      snapshots.push({ step, grid: grid.map(r => [...r]) });
    }
  }

  // Push final snapshot
  snapshots.push({ step: nsteps, grid: grid.map(r => [...r]) });

  return { grid: grid.map(r => [...r]), energyHistory, magnetizationHistory, snapshots };
}

export function IsingModel() {
  const [size, setSize] = useState(30);
  const [beta, setBeta] = useState(0.44);
  const [nsteps, setNsteps] = useState(5000);
  const [seed, setSeed] = useState(0);
  const { mergeLayout } = usePlotlyTheme();

  const result = useMemo(() => {
    // seed is used only as a trigger for re-run
    return runIsing(size, nsteps, beta, 4);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, nsteps, beta, seed]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Size: {size}</label>
          <Slider
            min={5}
            max={80}
            step={5}
            value={[size]}
            onValueChange={([v]) => setSize(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Beta (1/T): {beta.toFixed(2)}</label>
          <Slider
            min={0.01}
            max={2.0}
            step={0.01}
            value={[beta]}
            onValueChange={([v]) => setBeta(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Steps: {nsteps}</label>
          <Slider
            min={100}
            max={50000}
            step={100}
            value={[nsteps]}
            onValueChange={([v]) => setNsteps(v)}
            className="w-48"
          />
        </div>
        <button
          onClick={() => setSeed(s => s + 1)}
          className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4"
        >
          Re-run
        </button>
      </div>

      {/* Snapshots */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        {result.snapshots.map((snap, idx) => (
          <div key={idx}>
            <Plot
              data={[{
                z: snap.grid,
                type: 'heatmap',
                colorscale: [[0, '#1e3a5f'], [0.5, '#0f0f19'], [1, '#5f1e1e']],
                showscale: false,
                zmin: -1,
                zmax: 1,
              }]}
              layout={mergeLayout({
                title: { text: `Step ${snap.step}`, font: { size: 11 } },
                xaxis: { showticklabels: false },
                yaxis: { showticklabels: false },
                margin: { t: 30, r: 5, b: 5, l: 5 },
              })}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%', height: 180 }}
            />
          </div>
        ))}
      </div>

      {/* Energy and magnetization time series */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[
            {
              y: result.energyHistory,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#ef4444', width: 1 },
              name: 'Energy',
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Energy vs Time', font: { size: 13 } },
            xaxis: { title: { text: 'Timestep' } },
            yaxis: { title: { text: 'Energy' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 300 }}
        />
        <Plot
          data={[
            {
              y: result.magnetizationHistory.map(Math.abs),
              type: 'scatter',
              mode: 'lines',
              line: { color: '#f59e0b', width: 1 },
              name: '|Magnetization|',
            },
          ]}
          layout={mergeLayout({
            title: { text: '|Magnetization| vs Time', font: { size: 13 } },
            xaxis: { title: { text: 'Timestep' } },
            yaxis: { title: { text: '|M|' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 300 }}
        />
      </div>
    </div>
  );
}
