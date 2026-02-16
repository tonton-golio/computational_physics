'use client';

import React, { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Button } from '@/components/ui/button';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const SIZE = 50;

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
    setGrid(createRandomGrid());
  };

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <Button onClick={start} disabled={running}>Start</Button>
        <Button onClick={stop} disabled={!running}>Stop</Button>
        <Button onClick={reset}>Reset</Button>
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
        layout={{
          width: 600,
          height: 600,
          xaxis: { showticklabels: false },
          yaxis: { showticklabels: false },
          margin: { t: 0, b: 0, l: 0, r: 0 },
        }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}