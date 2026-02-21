'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';

/**
 * NotchDeltaCheckerboard: Lateral inhibition simulation (Notch-Delta signaling).
 *
 * Each cell has a Delta level D_i. Notch activation in cell i is proportional
 * to the average Delta of its neighbors. Notch represses Delta production in
 * the cell itself.
 *
 * dD_i/dt = v * f(D_neighbors) - gamma * D_i
 *
 * Where f is a repressive Hill function: f(S) = 1 / (1 + (S/K)^n)
 * and D_neighbors = average Delta of neighbors.
 *
 * This creates a checkerboard pattern: high-Delta cells next to low-Delta cells.
 */

const GRID_SIZE = 20;

function getNeighbors(row: number, col: number, size: number): Array<[number, number]> {
  const neighbors: Array<[number, number]> = [];
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  for (const [dr, dc] of dirs) {
    const r = row + dr;
    const c = col + dc;
    if (r >= 0 && r < size && c >= 0 && c < size) {
      neighbors.push([r, c]);
    }
  }
  return neighbors;
}

function initGrid(size: number, seed: number): Float64Array {
  let state = seed;
  const rand = () => {
    state = (state * 1664525 + 1013904223) & 0x7fffffff;
    return state / 0x7fffffff;
  };

  const grid = new Float64Array(size * size);
  for (let i = 0; i < grid.length; i++) {
    grid[i] = rand(); // Random initial Delta levels between 0 and 1
  }
  return grid;
}

function simulateStep(
  grid: Float64Array,
  size: number,
  n: number,
  K: number,
  v: number,
  gamma: number,
  dt: number,
): Float64Array {
  const newGrid = new Float64Array(grid.length);

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const idx = row * size + col;
      const neighbors = getNeighbors(row, col, size);

      // Average Delta from neighbors (Notch signal)
      let neighborSum = 0;
      for (const [nr, nc] of neighbors) {
        neighborSum += grid[nr * size + nc];
      }
      const avgNeighborDelta = neighborSum / neighbors.length;

      // Hill repression: high neighbor Delta -> low own Delta production
      const hillRepress = 1 / (1 + Math.pow(avgNeighborDelta / K, n));
      const production = v * hillRepress;
      const degradation = gamma * grid[idx];

      newGrid[idx] = Math.max(0, grid[idx] + (production - degradation) * dt);
    }
  }

  return newGrid;
}

function simulateToSteady(
  initialGrid: Float64Array,
  size: number,
  n: number,
  K: number,
  v: number,
  gamma: number,
  totalSteps: number,
): Float64Array {
  let grid = initialGrid;
  const dt = 0.05;

  for (let step = 0; step < totalSteps; step++) {
    grid = simulateStep(grid, size, n, K, v, gamma, dt);
  }

  return grid;
}

function deltaToColor(value: number, maxVal: number): string {
  const t = Math.min(value / Math.max(maxVal, 0.01), 1);
  // From dark (low Delta) to bright green (high Delta)
  const r = Math.round(20 + 20 * (1 - t));
  const g = Math.round(30 + 200 * t);
  const b = Math.round(40 + 30 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

export default function NotchDeltaCheckerboard() {
  const [hillN, setHillN] = useState(4);
  const [K, setK] = useState(0.5);
  const [steps, setSteps] = useState(200);
  const [seed, setSeed] = useState(42);

  const v = 1.0;
  const gamma = 1.0;

  const { grid, maxVal, patternScore } = useMemo(() => {
    const initial = initGrid(GRID_SIZE, seed);
    const grid = simulateToSteady(initial, GRID_SIZE, hillN, K, v, gamma, steps);

    const maxVal = Math.max(...Array.from(grid));

    // Compute a "checkerboard score": how well neighbors differ
    let totalDiff = 0;
    let count = 0;
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        const idx = row * GRID_SIZE + col;
        const neighbors = getNeighbors(row, col, GRID_SIZE);
        for (const [nr, nc] of neighbors) {
          totalDiff += Math.abs(grid[idx] - grid[nr * GRID_SIZE + nc]);
          count++;
        }
      }
    }
    const avgRange = maxVal > 0 ? maxVal : 1;
    const patternScore = totalDiff / count / avgRange;

    return { grid, maxVal, patternScore };
  }, [hillN, K, steps, seed]);

  const cellSize = 20;
  const svgSize = GRID_SIZE * cellSize;

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Notch-Delta Lateral Inhibition: Checkerboard Pattern
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Hill coefficient n: {hillN}
          </label>
          <Slider value={[hillN]} onValueChange={([v]) => setHillN(v)} min={1} max={8} step={1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Threshold K: {K.toFixed(2)}
          </label>
          <Slider value={[K]} onValueChange={([v]) => setK(v)} min={0.1} max={1.5} step={0.05} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Simulation steps: {steps}
          </label>
          <Slider value={[steps]} onValueChange={([v]) => setSteps(v)} min={10} max={500} step={10} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Random seed: {seed}
          </label>
          <Slider value={[seed]} onValueChange={([v]) => setSeed(v)} min={1} max={100} step={1} />
        </div>
      </div>

      <div className="mb-3 text-sm text-[var(--text-muted)]">
        Pattern score: <span className="font-mono font-semibold">{patternScore.toFixed(3)}</span>
        <span className="ml-2 text-xs">
          (higher = stronger checkerboard; 1.0 = perfect alternation)
        </span>
      </div>

      <div className="flex justify-center overflow-x-auto">
        <svg
          width={svgSize}
          height={svgSize}
          viewBox={`0 0 ${svgSize} ${svgSize}`}
          className="border border-[var(--border-strong)] rounded"
          style={{ maxWidth: '100%', height: 'auto' }}
        >
          {Array.from({ length: GRID_SIZE }, (_, row) =>
            Array.from({ length: GRID_SIZE }, (_, col) => {
              const idx = row * GRID_SIZE + col;
              const val = grid[idx];
              return (
                <rect
                  key={`${row}-${col}`}
                  x={col * cellSize}
                  y={row * cellSize}
                  width={cellSize}
                  height={cellSize}
                  fill={deltaToColor(val, maxVal)}
                  stroke="var(--surface-1)"
                  strokeWidth={0.5}
                />
              );
            })
          )}
        </svg>
      </div>

      {/* Color scale legend */}
      <div className="mt-3 flex items-center justify-center gap-2 text-xs text-[var(--text-muted)]">
        <span>Low Delta</span>
        <div
          className="h-3 rounded"
          style={{
            width: 120,
            background: `linear-gradient(to right, ${deltaToColor(0, 1)}, ${deltaToColor(0.5, 1)}, ${deltaToColor(1, 1)})`,
          }}
        />
        <span>High Delta</span>
      </div>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          Each cell&apos;s Delta production is repressed by the Notch signal from its neighbors&apos;
          Delta. The result: if your neighbor is loud (high Delta), you shut up (low Delta).
          With high Hill coefficient (n &ge; 3) and enough simulation steps, a striking
          checkerboard pattern emerges from random initial conditions.
          Try n = 1 to see how the pattern weakens without cooperativity &mdash; the cells can no
          longer make sharp decisions and the pattern becomes blurry.
        </p>
      </div>
    </div>
  );
}
