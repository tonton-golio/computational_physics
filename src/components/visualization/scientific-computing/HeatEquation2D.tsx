"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationToggle, SimulationPlayButton, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Animated 2D heat equation solver.
 * u_t = alpha * (u_xx + u_yy)
 * Uses explicit FTCS scheme with Dirichlet boundary conditions (u=0).
 * Initial conditions: hot spot in center, or two spots, or ring.
 */

type InitCondition = 'gaussian' | 'two-spots' | 'ring';

const GRID_SIZE = 80;

function createInitialCondition(type: InitCondition, n: number): number[][] {
  const u: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  const cx = n / 2;
  const cy = n / 2;
  const sigma = n / 8;

  switch (type) {
    case 'gaussian':
      for (let i = 1; i < n - 1; i++) {
        for (let j = 1; j < n - 1; j++) {
          const dx = i - cx;
          const dy = j - cy;
          u[i][j] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        }
      }
      break;
    case 'two-spots':
      for (let i = 1; i < n - 1; i++) {
        for (let j = 1; j < n - 1; j++) {
          const dx1 = i - n * 0.3;
          const dy1 = j - n * 0.3;
          const dx2 = i - n * 0.7;
          const dy2 = j - n * 0.7;
          u[i][j] =
            Math.exp(-(dx1 * dx1 + dy1 * dy1) / (2 * (sigma * 0.6) ** 2)) +
            0.7 * Math.exp(-(dx2 * dx2 + dy2 * dy2) / (2 * (sigma * 0.6) ** 2));
        }
      }
      break;
    case 'ring':
      for (let i = 1; i < n - 1; i++) {
        for (let j = 1; j < n - 1; j++) {
          const dx = i - cx;
          const dy = j - cy;
          const r = Math.sqrt(dx * dx + dy * dy);
          const rTarget = n * 0.25;
          u[i][j] = Math.exp(-((r - rTarget) ** 2) / (2 * (sigma * 0.3) ** 2));
        }
      }
      break;
  }
  return u;
}

function stepFTCS(u: number[][], alpha: number, dx: number, dt: number): number[][] {
  const n = u.length;
  const r = (alpha * dt) / (dx * dx);

  // Stability check: r must be <= 0.25 for 2D
  const rEff = Math.min(r, 0.24);

  const uNew: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => u[i][j]),
  );

  for (let i = 1; i < n - 1; i++) {
    for (let j = 1; j < n - 1; j++) {
      uNew[i][j] =
        u[i][j] +
        rEff * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - 4 * u[i][j]);
    }
  }
  return uNew;
}

export default function HeatEquation2D({}: SimulationComponentProps) {
  const [initType, setInitType] = useState<InitCondition>('gaussian');
  const [diffusivity, setDiffusivity] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const [stepsPerFrame, setStepsPerFrame] = useState(5);

  const gridRef = useRef<number[][]>(createInitialCondition('gaussian', GRID_SIZE));
  const [displayGrid, setDisplayGrid] = useState<number[][]>(() =>
    createInitialCondition('gaussian', GRID_SIZE),
  );
  const [timeStep, setTimeStep] = useState(0);
  const animRef = useRef<number | null>(null);

  const dx = 1 / GRID_SIZE;
  const dt = 0.2 * dx * dx / Math.max(diffusivity, 0.01); // Keep r < 0.25

  const reset = useCallback(() => {
    setPlaying(false);
    if (animRef.current) cancelAnimationFrame(animRef.current);
    const newGrid = createInitialCondition(initType, GRID_SIZE);
    gridRef.current = newGrid;
    setDisplayGrid(newGrid);
    setTimeStep(0);
  }, [initType]);

  // Reset when initial condition changes
  useEffect(() => {
    reset();
  }, [initType, reset]);

  // Animation loop
  useEffect(() => {
    if (!playing) {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      return;
    }

    const animate = () => {
      let grid = gridRef.current;
      for (let s = 0; s < stepsPerFrame; s++) {
        grid = stepFTCS(grid, diffusivity, dx, dt);
      }
      gridRef.current = grid;
      setDisplayGrid(grid);
      setTimeStep((prev) => prev + stepsPerFrame);
      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [playing, diffusivity, dx, dt, stepsPerFrame]);

  // Compute statistics
  const stats = React.useMemo(() => {
    let maxTemp = 0;
    let totalEnergy = 0;
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        const v = displayGrid[i][j];
        if (v > maxTemp) maxTemp = v;
        totalEnergy += v;
      }
    }
    return { maxTemp, totalEnergy, time: timeStep * dt };
  }, [displayGrid, timeStep, dt]);

  // Prepare heatmap data
  const heatmapZ = displayGrid;
  const xGrid = Array.from({ length: GRID_SIZE }, (_, i) => i * dx);
  const yGrid = Array.from({ length: GRID_SIZE }, (_, i) => i * dx);

  return (
    <SimulationPanel title="2D Heat Equation Solver" caption="Watch heat diffuse from an initial distribution. The FTCS scheme solves u_t = \u03B1(u_xx + u_yy) with zero boundary conditions.">
      <SimulationSettings>
        <div>
          <SimulationLabel>Initial condition</SimulationLabel>
          <SimulationToggle
            options={[
              { label: 'Gaussian', value: 'gaussian' },
              { label: 'Two spots', value: 'two-spots' },
              { label: 'Ring', value: 'ring' },
            ]}
            value={initType}
            onChange={(v) => setInitType(v as InitCondition)}
          />
        </div>
        <div className="flex items-end gap-2">
          <SimulationPlayButton
            isRunning={playing}
            onToggle={() => setPlaying(p => !p)}
          />
          <SimulationButton
            variant="secondary"
            onClick={reset}
          >
            Reset
          </SimulationButton>
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <SimulationLabel>Diffusivity &alpha;: {diffusivity.toFixed(2)}</SimulationLabel>
            <Slider
              value={[diffusivity]}
              onValueChange={([v]) => setDiffusivity(v)}
              min={0.05}
              max={2.0}
              step={0.05}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Speed (steps/frame): {stepsPerFrame}</SimulationLabel>
            <Slider
              value={[stepsPerFrame]}
              onValueChange={([v]) => setStepsPerFrame(v)}
              min={1}
              max={20}
              step={1}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasHeatmap
          data={[
            {
              z: heatmapZ,
              x: xGrid,
              y: yGrid,
              colorscale: 'Inferno',
              showscale: true,
              zmin: 0,
              zmax: 1,
            },
          ]}
          layout={{
            title: { text: `Temperature at t = ${stats.time.toFixed(4)}` },
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'y' } },
            margin: { t: 40, r: 60, b: 50, l: 60 },
          }}
          style={{ width: '100%', height: 450 }}
        />
      </SimulationMain>

      <SimulationResults>
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Max temperature</div>
            <div className="text-sm font-mono font-semibold text-[var(--accent)]">
              {stats.maxTemp.toFixed(4)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Total energy</div>
            <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
              {stats.totalEnergy.toFixed(2)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Time steps</div>
            <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
              {timeStep}
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
