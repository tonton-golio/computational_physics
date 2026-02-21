'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { useTheme } from '@/lib/use-theme';

const SIZE = 50;

// ---------------------------------------------------------------------------
// Game Logic
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// p5.js 2D Canvas
// ---------------------------------------------------------------------------

interface CanvasProps {
  gridRef: React.RefObject<number[][] | null>;
  isDark: boolean;
  onCellToggle: (i: number, j: number) => void;
}

function GameOfLifeCanvas({ gridRef, isDark, onCellToggle }: CanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const p5Ref = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    let cancelled = false;

    import('p5').then(({ default: p5 }) => {
      if (cancelled || !containerRef.current) return;

      if (p5Ref.current) {
        p5Ref.current.remove();
        p5Ref.current = null;
      }

      const instance = new p5((p: any) => {
        const canvasWidth = containerRef.current?.clientWidth || 600;
        const cellSize = canvasWidth / SIZE;

        p.setup = () => {
          p.createCanvas(canvasWidth, canvasWidth);
          p.pixelDensity(1);
          p.frameRate(30);
        };

        p.draw = () => {
          const grid = gridRef.current;
          if (!grid) return;

          if (isDark) p.background(7, 7, 16);
          else p.background(240, 244, 255);

          const ctx = p.drawingContext as CanvasRenderingContext2D;
          p.noStroke();

          // Dead cells
          if (isDark) p.fill(18, 18, 36);
          else p.fill(223, 232, 251);
          for (let i = 0; i < SIZE; i++) {
            for (let j = 0; j < SIZE; j++) {
              if (grid[i][j] === 0) {
                p.rect(j * cellSize + 0.5, i * cellSize + 0.5, cellSize - 1, cellSize - 1, 1);
              }
            }
          }

          // Alive cells with glow in dark mode
          if (isDark) {
            ctx.shadowColor = 'rgba(0, 255, 204, 0.6)';
            ctx.shadowBlur = 8;
            p.fill(0, 255, 204);
          } else {
            p.fill(0, 170, 136);
          }
          for (let i = 0; i < SIZE; i++) {
            for (let j = 0; j < SIZE; j++) {
              if (grid[i][j] === 1) {
                p.rect(j * cellSize + 0.5, i * cellSize + 0.5, cellSize - 1, cellSize - 1, 1);
              }
            }
          }
          if (isDark) {
            ctx.shadowBlur = 0;
            ctx.shadowColor = 'transparent';
          }
        };

        p.mousePressed = () => {
          if (p.mouseX >= 0 && p.mouseX < canvasWidth && p.mouseY >= 0 && p.mouseY < canvasWidth) {
            const j = Math.floor(p.mouseX / cellSize);
            const i = Math.floor(p.mouseY / cellSize);
            if (i >= 0 && i < SIZE && j >= 0 && j < SIZE) {
              onCellToggle(i, j);
            }
          }
        };
      }, containerRef.current);

      p5Ref.current = instance;
    });

    return () => {
      cancelled = true;
      if (p5Ref.current) {
        p5Ref.current.remove();
        p5Ref.current = null;
      }
    };
  }, [isDark, gridRef, onCellToggle]);

  return <div ref={containerRef} className="w-full" />;
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export function GameOfLife() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [grid, setGrid] = useState(createRandomGrid);
  const [running, setRunning] = useState(false);
  const [selectedPattern, setSelectedPattern] = useState('random');
  const [speed, setSpeed] = useState(200);
  const [generation, setGeneration] = useState(0);

  const gridRef = useRef<number[][] | null>(grid);
  useEffect(() => { gridRef.current = grid; }, [grid]);

  const selectedPatternRef = useRef(selectedPattern);
  useEffect(() => {
    selectedPatternRef.current = selectedPattern;
  }, [selectedPattern]);

  useEffect(() => {
    if (!running) return;

    const interval = setInterval(() => {
      setGrid(currentGrid => nextGeneration(currentGrid));
      setGeneration(g => g + 1);
    }, speed);

    return () => clearInterval(interval);
  }, [running, speed]);

  const population = grid.reduce((sum, row) => sum + row.reduce((s, c) => s + c, 0), 0);

  const start = useCallback(() => setRunning(true), []);
  const stop = useCallback(() => setRunning(false), []);

  const reset = useCallback(() => {
    setRunning(false);
    setGeneration(0);
    const pat = selectedPatternRef.current;
    if (pat === 'random') {
      setGrid(createRandomGrid());
    } else {
      const emptyGrid = createEmptyGrid();
      const p = patterns[pat as keyof typeof patterns];
      if (Array.isArray(p)) {
        const startX = Math.floor(SIZE / 2) - Math.floor(p.length / 2);
        const startY = Math.floor(SIZE / 2) - Math.floor(p[0].length / 2);
        setGrid(setPattern(emptyGrid, p, startX, startY));
      } else {
        setGrid(p());
      }
    }
  }, []);

  const selectPattern = useCallback((pattern: string) => {
    setSelectedPattern(pattern);
    selectedPatternRef.current = pattern;
    setRunning(false);
    setGeneration(0);
    if (pattern === 'random') {
      setGrid(createRandomGrid());
    } else {
      const emptyGrid = createEmptyGrid();
      const p = patterns[pattern as keyof typeof patterns];
      if (Array.isArray(p)) {
        const startX = Math.floor(SIZE / 2) - Math.floor(p.length / 2);
        const startY = Math.floor(SIZE / 2) - Math.floor(p[0].length / 2);
        setGrid(setPattern(emptyGrid, p, startX, startY));
      } else {
        setGrid(p());
      }
    }
  }, []);

  const toggleCell = useCallback((i: number, j: number) => {
    setGrid(g => {
      const newGrid = g.map(row => [...row]);
      newGrid[i][j] = newGrid[i][j] === 1 ? 0 : 1;
      return newGrid;
    });
  }, []);

  return (
    <div>
      {/* Controls */}
      <div className="mb-4 flex gap-2 flex-wrap items-center">
        <Button onClick={start} disabled={running}>Play</Button>
        <Button onClick={stop} disabled={!running}>Pause</Button>
        <Button onClick={reset}>Reset</Button>
        <span className="text-sm text-[var(--text-muted)] ml-2">
          Gen: {generation} | Pop: {population}
        </span>
      </div>

      <div className="mb-4 flex gap-4 flex-wrap items-center">
        <div>
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
        <div className="flex items-center gap-2">
          <label className="text-[var(--text-strong)]">Speed:</label>
          <input
            type="range"
            min={50}
            max={500}
            step={10}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-32 accent-[var(--accent)]"
          />
          <span className="text-[var(--text-muted)] text-sm w-16">{speed}ms</span>
        </div>
      </div>

      {/* 2D Canvas */}
      <div
        className="w-full rounded-lg overflow-hidden"
        style={{ background: isDark ? '#070710' : '#f0f4ff' }}
      >
        <GameOfLifeCanvas gridRef={gridRef} isDark={isDark} onCellToggle={toggleCell} />
      </div>

      <p className="text-xs text-[var(--text-muted)] mt-2">Click on cells to toggle them.</p>
    </div>
  );
}
