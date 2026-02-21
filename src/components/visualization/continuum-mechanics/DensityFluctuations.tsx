'use client';

import React, { useRef, useEffect, useCallback, useState } from 'react';

/**
 * Visualises the continuum approximation by showing the same field of
 * randomly-placed particles at three zoom levels.  At the smallest scale
 * density fluctuates wildly; at a macroscopic scale it smooths out,
 * justifying the continuum assumption.
 */

const PARTICLE_COUNT = 600;
const COLORS = {
  particle: '#60a5fa',
  gridLine: 'rgba(148,163,184,0.18)',
  densityHigh: '#ef4444',
  densityMid: '#f59e0b',
  densityLow: '#3b82f6',
  text: '#cbd5e1',
};

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

interface PanelProps {
  particles: { x: number; y: number }[];
  gridN: number;
  label: string;
  subtitle: string;
}

function DensityPanel({ particles, gridN, label, subtitle }: PanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = canvas.width;
    const dpr = window.devicePixelRatio || 1;
    const cssSize = size / dpr;

    ctx.clearRect(0, 0, size, size);

    // Compute density per cell
    const cellSize = 1 / gridN;
    const counts = Array.from({ length: gridN * gridN }, () => 0);
    for (const p of particles) {
      const col = Math.min(gridN - 1, Math.floor(p.x / cellSize));
      const row = Math.min(gridN - 1, Math.floor(p.y / cellSize));
      counts[row * gridN + col]++;
    }
    const maxCount = Math.max(...counts, 1);

    // Draw density-coloured cells
    const pxCell = cssSize / gridN;
    for (let r = 0; r < gridN; r++) {
      for (let c = 0; c < gridN; c++) {
        const frac = counts[r * gridN + c] / maxCount;
        // Map fraction to colour
        let color: string;
        if (frac > 0.7) color = COLORS.densityHigh;
        else if (frac > 0.35) color = COLORS.densityMid;
        else color = COLORS.densityLow;
        ctx.globalAlpha = 0.12 + frac * 0.22;
        ctx.fillStyle = color;
        ctx.fillRect(c * pxCell * dpr, r * pxCell * dpr, pxCell * dpr, pxCell * dpr);
      }
    }
    ctx.globalAlpha = 1;

    // Grid lines
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = dpr;
    for (let i = 0; i <= gridN; i++) {
      const pos = i * pxCell * dpr;
      ctx.beginPath();
      ctx.moveTo(pos, 0);
      ctx.lineTo(pos, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, pos);
      ctx.lineTo(size, pos);
      ctx.stroke();
    }

    // Particles
    ctx.fillStyle = COLORS.particle;
    const radius = Math.max(1.2, 2.5 * dpr);
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x * cssSize * dpr, p.y * cssSize * dpr, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [particles, gridN]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssSize = 180;
    canvas.width = cssSize * dpr;
    canvas.height = cssSize * dpr;
    canvas.style.width = `${cssSize}px`;
    canvas.style.height = `${cssSize}px`;
    draw();
  }, [draw]);

  return (
    <div className="flex flex-col items-center gap-2">
      <canvas ref={canvasRef} className="rounded-md border border-[var(--border-strong)]" />
      <div className="text-center">
        <div className="text-sm font-medium text-[var(--text-strong)]">{label}</div>
        <div className="text-xs text-[var(--text-muted)]">{subtitle}</div>
      </div>
    </div>
  );
}

export default function DensityFluctuations() {
  const [seed] = useState(42);

  const particles = React.useMemo(() => {
    const rng = seededRandom(seed);
    return Array.from({ length: PARTICLE_COUNT }, () => ({
      x: rng(),
      y: rng(),
    }));
  }, [seed]);

  return (
    <div className="w-full rounded-lg p-6 mb-8">
      <h3 className="text-lg font-semibold mb-2 text-[var(--text-strong)]">
        The Continuum Approximation
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-5">
        The same particle field viewed at three different coarsening scales.
        At fine resolution density fluctuates from cell to cell; as the
        averaging window grows the fluctuations vanish and a smooth density
        field emerges &mdash; the continuum limit.
      </p>
      <div className="flex flex-wrap justify-center gap-6">
        <DensityPanel particles={particles} gridN={20} label="Fine scale" subtitle="Large fluctuations" />
        <DensityPanel particles={particles} gridN={8} label="Intermediate" subtitle="Fluctuations reduce" />
        <DensityPanel particles={particles} gridN={3} label="Continuum limit" subtitle="Smooth density" />
      </div>
      <div className="mt-4 flex justify-center gap-4 text-xs text-[var(--text-soft)]">
        <span><span className="inline-block w-2.5 h-2.5 rounded-full mr-1 align-middle" style={{ background: COLORS.densityHigh }} />High density</span>
        <span><span className="inline-block w-2.5 h-2.5 rounded-full mr-1 align-middle" style={{ background: COLORS.densityMid }} />Medium</span>
        <span><span className="inline-block w-2.5 h-2.5 rounded-full mr-1 align-middle" style={{ background: COLORS.densityLow }} />Low density</span>
      </div>
    </div>
  );
}
