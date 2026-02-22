"use client";

import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { getCanvasTheme } from '@/lib/canvas-theme';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const TRACE_COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899',
];

/**
 * Interactive 2x2 condition number amplifier.
 *
 * The user controls the angle between two column vectors and a
 * perturbation size epsilon. The canvas draws:
 *   - Column vectors a1, a2 (colored arrows)
 *   - RHS vector b and solution x
 *   - Perturbed RHS b+db and perturbed solution x+dx
 *
 * Live readout: cond(A), ||dx||/||x||, ||db||/||b||, amplification ratio.
 */
export default function ConditionNumberDemo({}: SimulationComponentProps) {
  const [angleDeg, setAngleDeg] = useState(30);
  const [epsilon, setEpsilon] = useState(0.05);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const angleRad = (angleDeg * Math.PI) / 180;

  // Build 2x2 matrix from angle between columns
  const result = useMemo(() => {
    // Column 1 always points along (1, 0)
    const a1x = 1, a1y = 0;
    // Column 2 at angle from column 1
    const a2x = Math.cos(angleRad);
    const a2y = Math.sin(angleRad);

    const det = a1x * a2y - a2x * a1y;

    // Inverse
    const Ainv = [
      [a2y / det, -a2x / det],
      [-a1y / det, a1x / det],
    ];

    // Condition number (2-norm): use singular values
    // For 2x2: sigma = sqrt(eigenvalues of A^T A)
    const AtA00 = a1x * a1x + a1y * a1y;
    const AtA01 = a1x * a2x + a1y * a2y;
    const AtA11 = a2x * a2x + a2y * a2y;

    const trace = AtA00 + AtA11;
    const detAtA = AtA00 * AtA11 - AtA01 * AtA01;
    const disc = Math.sqrt(Math.max(0, trace * trace - 4 * detAtA));

    const sig1 = Math.sqrt((trace + disc) / 2);
    const sig2 = Math.sqrt(Math.max(0, (trace - disc) / 2));
    const condA = sig2 > 1e-15 ? sig1 / sig2 : Infinity;

    // RHS vector b = A * [1, 1]^T
    const bx = a1x + a2x;
    const by = a1y + a2y;

    // Solution x = [1, 1]
    const xx = 1, xy = 1;

    // Perturbed b: rotate b by a small angle to get db perpendicular-ish
    const bNorm = Math.sqrt(bx * bx + by * by);
    const dbx = -by / bNorm * epsilon * bNorm;
    const dby = bx / bNorm * epsilon * bNorm;

    const bpx = bx + dbx;
    const bpy = by + dby;

    // Perturbed solution: x_p = A^{-1} * b_p
    const xpx = Ainv[0][0] * bpx + Ainv[0][1] * bpy;
    const xpy = Ainv[1][0] * bpx + Ainv[1][1] * bpy;

    const dxx = xpx - xx;
    const dxy = xpy - xy;

    const xNorm = Math.sqrt(xx * xx + xy * xy);
    const dxNorm = Math.sqrt(dxx * dxx + dxy * dxy);
    const dbNorm = Math.sqrt(dbx * dbx + dby * dby);

    const relErrX = dxNorm / xNorm;
    const relErrB = dbNorm / bNorm;
    const amplification = relErrB > 1e-15 ? relErrX / relErrB : 0;

    return {
      a1: [a1x, a1y],
      a2: [a2x, a2y],
      b: [bx, by],
      bp: [bpx, bpy],
      x: [xx, xy],
      xp: [xpx, xpy],
      condA,
      relErrX,
      relErrB,
      amplification,
    };
  }, [angleRad, epsilon]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const theme = getCanvasTheme();

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;

    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    // Two panels: left = column space (b, b'), right = solution space (x, x')
    const midX = W / 2;
    const padX = 40;
    const padY = 30;

    // Left panel center
    const lcx = padX + (midX - padX) / 2;
    const lcy = padY + (H - 2 * padY) / 2;
    const leftScale = Math.min(midX - 2 * padX, H - 2 * padY) / 5;

    // Right panel center
    const rcx = midX + padX + (midX - 2 * padX) / 2;
    const rcy = lcy;

    // Determine right panel scale: make vectors fit
    const maxXcoord = Math.max(
      Math.abs(result.x[0]), Math.abs(result.x[1]),
      Math.abs(result.xp[0]), Math.abs(result.xp[1]),
    );
    const rightScale = maxXcoord > 0.1
      ? Math.min(midX - 2 * padX, H - 2 * padY) / (2.5 * maxXcoord)
      : leftScale;

    // Draw grid and axes helper
    const drawAxes = (cx: number, cy: number, scale: number, label: string) => {
      ctx.strokeStyle = theme.grid;
      ctx.lineWidth = 0.5;
      // Grid lines
      for (let i = -3; i <= 3; i++) {
        ctx.beginPath();
        ctx.moveTo(cx + i * scale, cy - 3 * scale);
        ctx.lineTo(cx + i * scale, cy + 3 * scale);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - 3 * scale, cy + i * scale);
        ctx.lineTo(cx + 3 * scale, cy + i * scale);
        ctx.stroke();
      }
      // Axes
      ctx.strokeStyle = theme.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - 3 * scale, cy);
      ctx.lineTo(cx + 3 * scale, cy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx, cy - 3 * scale);
      ctx.lineTo(cx, cy + 3 * scale);
      ctx.stroke();
      // Label
      ctx.fillStyle = theme.text;
      ctx.font = `bold 13px ${'ui-monospace, SFMono-Regular, Menlo, monospace'}`;
      ctx.textAlign = 'center';
      ctx.fillText(label, cx, padY - 8);
    };

    const drawArrow = (
      cx: number, cy: number, scale: number,
      vx: number, vy: number,
      color: string, label: string, lineWidth = 2,
    ) => {
      const ex = cx + vx * scale;
      const ey = cy - vy * scale; // flip y
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      // Arrowhead
      const len = Math.sqrt(vx * vx + vy * vy) * scale;
      if (len < 3) return;
      const headLen = Math.min(10, len * 0.2);
      const angle = Math.atan2(-(vy), vx); // account for flipped y
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(
        ex - headLen * Math.cos(angle - Math.PI / 6),
        ey + headLen * Math.sin(angle - Math.PI / 6),
      );
      ctx.lineTo(
        ex - headLen * Math.cos(angle + Math.PI / 6),
        ey + headLen * Math.sin(angle + Math.PI / 6),
      );
      ctx.closePath();
      ctx.fill();

      // Label
      ctx.fillStyle = color;
      ctx.font = `bold 12px ${'ui-monospace, SFMono-Regular, Menlo, monospace'}`;
      ctx.textAlign = 'left';
      ctx.fillText(label, ex + 6, ey - 6);
    };

    // Left: Column space (b vectors)
    drawAxes(lcx, lcy, leftScale, 'Column Space (b)');

    // Draw column vectors (dim)
    drawArrow(lcx, lcy, leftScale, result.a1[0], result.a1[1], TRACE_COLORS[4] ?? '#888', 'a1', 1.5);
    drawArrow(lcx, lcy, leftScale, result.a2[0], result.a2[1], TRACE_COLORS[5] ?? '#888', 'a2', 1.5);

    // Draw b
    drawArrow(lcx, lcy, leftScale, result.b[0], result.b[1], TRACE_COLORS[0], 'b', 2.5);
    // Draw b'
    drawArrow(lcx, lcy, leftScale, result.bp[0], result.bp[1], TRACE_COLORS[1], "b'", 2.5);

    // Draw db as dashed
    const bEndX = lcx + result.b[0] * leftScale;
    const bEndY = lcy - result.b[1] * leftScale;
    const bpEndX = lcx + result.bp[0] * leftScale;
    const bpEndY = lcy - result.bp[1] * leftScale;
    ctx.strokeStyle = TRACE_COLORS[1];
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(bEndX, bEndY);
    ctx.lineTo(bpEndX, bpEndY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Right: Solution space (x vectors)
    drawAxes(rcx, rcy, rightScale, 'Solution Space (x)');

    // Draw x
    drawArrow(rcx, rcy, rightScale, result.x[0], result.x[1], TRACE_COLORS[2], 'x', 2.5);
    // Draw x'
    drawArrow(rcx, rcy, rightScale, result.xp[0], result.xp[1], TRACE_COLORS[3], "x'", 2.5);

    // Draw dx as dashed
    const xEndX = rcx + result.x[0] * rightScale;
    const xEndY = rcy - result.x[1] * rightScale;
    const xpEndX = rcx + result.xp[0] * rightScale;
    const xpEndY = rcy - result.xp[1] * rightScale;
    ctx.strokeStyle = TRACE_COLORS[3];
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(xEndX, xEndY);
    ctx.lineTo(xpEndX, xpEndY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Divider
    ctx.strokeStyle = theme.grid;
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(midX, padY - 20);
    ctx.lineTo(midX, H);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [result]);

  useEffect(() => {
    draw();
    const handleResize = () => draw();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [draw]);

  return (
    <SimulationPanel title="Condition Number Amplifier" caption="A small perturbation in the right-hand side b is amplified in the solution x. The amplification factor is bounded by cond(A). Narrow the angle between column vectors to see the condition number explode.">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <SimulationLabel>
              Angle between columns: {angleDeg}°
            </SimulationLabel>
            <Slider
              value={[angleDeg]}
              onValueChange={(val) => setAngleDeg(val[0])}
              min={1}
              max={90}
              step={1}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>
              Perturbation size (epsilon): {epsilon.toFixed(3)}
            </SimulationLabel>
            <Slider
              value={[epsilon]}
              onValueChange={(val) => setEpsilon(val[0])}
              min={0.001}
              max={0.2}
              step={0.001}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain scaleMode="contain">
        <canvas
          ref={canvasRef}
          className="w-full rounded-md border border-[var(--border)]"
          style={{ height: 320 }}
        />
      </SimulationMain>

      <SimulationResults>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">cond(A)</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {result.condA < 1e6 ? result.condA.toFixed(1) : result.condA.toExponential(1)}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">||db||/||b||</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {(result.relErrB * 100).toFixed(2)}%
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">||dx||/||x||</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {(result.relErrX * 100).toFixed(2)}%
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Amplification</div>
            <div className="text-base font-mono font-semibold text-[var(--accent)]">
              {result.amplification.toFixed(1)}x
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
