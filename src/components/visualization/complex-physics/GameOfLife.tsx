'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Button } from '@/components/ui/button';
import { usePlotlyTheme } from '@/lib/plotly-theme';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const SIZE = 50;

function createEmptyGrid(): number[][] {
  return Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
}

function createRandomGrid(): number[][] {
  const grid = [];
  for (let i = 0; i < SIZE; i++) {
    const row = [];
    for (let j = 0; j < SIZE; j++) {
      row.push(Math.random() > 0.7 ? 1 : 0);
    }
    grid.push(row);
  }
  return grid;
}

function setPattern(grid: number[][], pattern: number[][], startX: number, startY: number): number[][] {
  const newGrid = grid.map(row => [...row]);
  pattern.forEach((row, i) => {
    row.forEach((cell, j) => {
      if (startX + i < SIZE && startY + j < SIZE) {
        newGrid[startX + i][startY + j] = cell;
      }
    });
  });
  return newGrid;
}

const patterns = {
  random: createRandomGrid,
  glider: [
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
  ],
  blinker: [
    [1, 1, 1],
  ],
  beacon: [
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
  ],
  toad: [
    [0, 1, 1, 1],
    [1, 1, 1, 0],
  ],
  block: [
    [1, 1],
    [1, 1],
  ],
  beehive: [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
  ],
};

function countNeighbors(grid: number[][], x: number, y: number): number {
  let count = 0;
  for (let i = -1; i <= 1; i++) {
    for (let j = -1; j <= 1; j++) {
      if (i === 0 && j === 0) continue;
      const ni = x + i;
      const nj = y + j;
      if (ni >= 0 && ni < SIZE && nj >= 0 && nj < SIZE) {
        count += grid[ni][nj];
      }
    }
  }
  return count;
}

function nextGeneration(grid: number[][]): number[][] {
  const newGrid = grid.map(row => [...row]);
  for (let i = 0; i < SIZE; i++) {
    for (let j = 0; j < SIZE; j++) {
      const neighbors = countNeighbors(grid, i, j);
      if (grid[i][j] === 1) {
        if (neighbors < 2 || neighbors > 3) {
          newGrid[i][j] = 0;
        }
      } else {
        if (neighbors === 3) {
          newGrid[i][j] = 1;
        }
      }
    }
  }
  return newGrid;
}

export function GameOfLife() {
  const [grid, setGrid] = useState(createRandomGrid);
  const [running, setRunning] = useState(false);
  const [selectedPattern, setSelectedPattern] = useState('random');
  const { mergeLayout } = usePlotlyTheme();

  useEffect(() => {
    if (!running) return;

    const interval = setInterval(() => {
      setGrid(currentGrid => nextGeneration(currentGrid));
    }, 200);

    return () => clearInterval(interval);
  }, [running]);

  const start = () => setRunning(true);
  const stop = () => setRunning(false);
  const reset = () => {
    setRunning(false);
    if (selectedPattern === 'random') {
      setGrid(createRandomGrid());
    } else {
      const emptyGrid = createEmptyGrid();
      const pattern = patterns[selectedPattern as keyof typeof patterns];
      if (Array.isArray(pattern)) {
        const startX = Math.floor(SIZE / 2) - Math.floor(pattern.length / 2);
        const startY = Math.floor(SIZE / 2) - Math.floor(pattern[0].length / 2);
        setGrid(setPattern(emptyGrid, pattern, startX, startY));
      } else {
        setGrid(pattern());
      }
    }
  };

  const selectPattern = (pattern: string) => {
    setSelectedPattern(pattern);
    reset();
  };

  return (
    <div>
      <div className="mb-4 flex gap-2 flex-wrap">
        <Button onClick={start} disabled={running}>Play</Button>
        <Button onClick={stop} disabled={!running}>Pause</Button>
        <Button onClick={reset}>Reset</Button>
      </div>
      <div className="mb-4">
        <label className="text-[var(--text-strong)] mr-2">Initial Pattern:</label>
        <select
          value={selectedPattern}
          onChange={(e) => selectPattern(e.target.value)}
          className="bg-[var(--surface-3)] text-[var(--text-strong)] p-2 rounded border border-[var(--border-strong)]"
        >
          <option value="random">Random</option>
          <option value="glider">Glider</option>
          <option value="blinker">Blinker</option>
          <option value="beacon">Beacon</option>
          <option value="toad">Toad</option>
          <option value="block">Block</option>
          <option value="beehive">Beehive</option>
        </select>
      </div>
      <Plot
        data={[
          {
            z: grid,
            type: 'heatmap',
            colorscale: [[0, 'white'], [1, 'black']],
            showscale: false,
          },
        ]}
        layout={mergeLayout({
          width: 600,
          height: 600,
          xaxis: { showticklabels: false },
          yaxis: { showticklabels: false },
          margin: { t: 0, b: 0, l: 0, r: 0 },
        })}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}