"use client";

import { useEffect, useMemo, useRef, useState } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationAux, SimulationLabel, SimulationButton } from '@/components/ui/simulation-panel';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { useTheme } from '@/lib/use-theme';

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function firstReturn1D(maxSteps: number, trials: number, rand: () => number): number[] {
  const out: number[] = [];
  for (let t = 0; t < trials; t++) {
    let x = 0;
    let returned = false;
    for (let step = 1; step <= maxSteps; step++) {
      x += rand() < 0.5 ? -1 : 1;
      if (x === 0) {
        out.push(step);
        returned = true;
        break;
      }
    }
    if (!returned) out.push(maxSteps + 1);
  }
  return out;
}

function firstReturn2D(maxSteps: number, trials: number, rand: () => number): number[] {
  const out: number[] = [];
  for (let t = 0; t < trials; t++) {
    let x = 0;
    let y = 0;
    let escaped = false;
    let returned = false;
    for (let step = 1; step <= maxSteps; step++) {
      const theta = rand() * 2 * Math.PI;
      x += Math.cos(theta);
      y += Math.sin(theta);
      const r = Math.sqrt(x * x + y * y);
      if (r > 1.5) escaped = true;
      if (escaped && r <= 1.0) {
        out.push(step);
        returned = true;
        break;
      }
    }
    if (!returned) out.push(maxSteps + 1);
  }
  return out;
}

function randomWalkPath2D(steps: number, rand: () => number): { x: number[]; y: number[] } {
  const x = [0];
  const y = [0];
  for (let i = 0; i < steps; i++) {
    const theta = rand() * 2 * Math.PI;
    x.push(x[x.length - 1] + Math.cos(theta));
    y.push(y[y.length - 1] + Math.sin(theta));
  }
  return { x, y };
}

/* ---------- p5.js animated walk trail ---------- */

function WalkCanvas({ path, pathSteps, isDark }: { path: { x: number[]; y: number[] }; pathSteps: number; isDark: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const p5Ref = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    let instance: any;

    import('p5').then(({ default: p5 }) => {
      if (!containerRef.current) return;
      if (p5Ref.current) {
        p5Ref.current.remove();
        p5Ref.current = null;
      }

      instance = new p5((p: any) => {
        const canvasSize = containerRef.current?.clientWidth || 500;
        let frame = 0;
        const totalPoints = path.x.length;
        const animSpeed = Math.max(1, Math.floor(totalPoints / 120));

        const xs = path.x;
        const ys = path.y;
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (let i = 0; i < xs.length; i++) {
          if (xs[i] < minX) minX = xs[i];
          if (xs[i] > maxX) maxX = xs[i];
          if (ys[i] < minY) minY = ys[i];
          if (ys[i] > maxY) maxY = ys[i];
        }
        const pad = Math.max(maxX - minX, maxY - minY) * 0.1 + 2;
        const rangeX = maxX - minX + 2 * pad;
        const rangeY = maxY - minY + 2 * pad;
        const range = Math.max(rangeX, rangeY);
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;

        function toScreen(wx: number, wy: number): [number, number] {
          return [
            (wx - cx + range / 2) / range * canvasSize,
            (-(wy - cy) + range / 2) / range * canvasSize,
          ];
        }

        p.setup = () => {
          p.createCanvas(canvasSize, canvasSize);
          p.pixelDensity(1);
          if (isDark) p.background(10, 10, 15);
          else p.background(240, 244, 255);
          p.frameRate(60);
        };

        p.draw = () => {
          if (isDark) p.background(10, 10, 15);
          else p.background(240, 244, 255);

          const drawUpTo = Math.min(frame, totalPoints - 1);

          // Glow layer
          p.noFill();
          for (let i = 1; i <= drawUpTo; i++) {
            const t = i / (totalPoints - 1);
            const [x1, y1] = toScreen(path.x[i - 1], path.y[i - 1]);
            const [x2, y2] = toScreen(path.x[i], path.y[i]);

            const r = p.lerp(0, 255, t);
            const g = p.lerp(255, 136, t);
            const b = p.lerp(204, 68, t);

            p.stroke(r, g, b, 50);
            p.strokeWeight(4.5 + t * 1.5);
            p.line(x1, y1, x2, y2);
          }

          // Main trail
          for (let i = 1; i <= drawUpTo; i++) {
            const t = i / (totalPoints - 1);
            const [x1, y1] = toScreen(path.x[i - 1], path.y[i - 1]);
            const [x2, y2] = toScreen(path.x[i], path.y[i]);

            const r = p.lerp(0, 255, t);
            const g = p.lerp(255, 136, t);
            const b = p.lerp(204, 68, t);

            p.stroke(r, g, b, 210);
            p.strokeWeight(1.2 + t * 1.0);
            p.line(x1, y1, x2, y2);
          }

          // Head dot
          if (drawUpTo > 0) {
            const tHead = drawUpTo / (totalPoints - 1);
            const [hx, hy] = toScreen(path.x[drawUpTo], path.y[drawUpTo]);
            const rH = p.lerp(0, 255, tHead);
            const gH = p.lerp(255, 136, tHead);
            const bH = p.lerp(204, 68, tHead);
            p.noStroke();
            p.fill(rH, gH, bH, 120);
            p.ellipse(hx, hy, 8, 8);
            p.fill(rH, gH, bH, 255);
            p.ellipse(hx, hy, 4, 4);
          }

          // Origin marker
          const [ox, oy] = toScreen(0, 0);
          p.noStroke();
          p.fill(255, 68, 68, 50);
          p.ellipse(ox, oy, 16, 16);
          p.fill(255, 68, 68, 190);
          p.ellipse(ox, oy, 9, 9);

          frame += animSpeed;
          if (frame >= totalPoints) {
            frame = totalPoints - 1;
            p.noLoop();
          }
        };
      }, containerRef.current);

      p5Ref.current = instance;
    });

    return () => {
      if (p5Ref.current) {
        p5Ref.current.remove();
        p5Ref.current = null;
      }
    };
  }, [path, pathSteps, isDark]);

  return <div ref={containerRef} className="w-full" />;
}

/* ---------- Main component ---------- */

export default function RandomWalkFirstReturn({}: SimulationComponentProps) {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [maxSteps, setMaxSteps] = useState(1200);
  const [trials, setTrials] = useState(700);
  const [pathSteps, setPathSteps] = useState(240);
  const [rerun, setRerun] = useState(0);

  const { ret1d, ret2d, path } = useMemo(() => {
    const rand = mulberry32((maxSteps * 17 + trials * 23 + pathSteps * 31 + rerun * 101) >>> 0);
    const a = firstReturn1D(maxSteps, trials, rand);
    const b = firstReturn2D(maxSteps, trials, rand);
    const c = randomWalkPath2D(pathSteps, rand);
    return { ret1d: a, ret2d: b, path: c };
  }, [maxSteps, trials, pathSteps, rerun]);

  const cutoff = maxSteps + 1;
  const finite1d = ret1d.filter(v => v < cutoff);
  const finite2d = ret2d.filter(v => v < cutoff);

  return (
    <SimulationPanel title="Random Walk First Return">
      <SimulationSettings>
        <SimulationButton variant="primary" onClick={() => setRerun(v => v + 1)}>
          Re-run
        </SimulationButton>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>Max Steps: {maxSteps}</SimulationLabel>
          <Slider value={[maxSteps]} onValueChange={([v]) => setMaxSteps(v)} min={200} max={3000} step={100} />
        </div>
        <div>
          <SimulationLabel>Trials: {trials}</SimulationLabel>
          <Slider value={[trials]} onValueChange={([v]) => setTrials(v)} min={100} max={3000} step={100} />
        </div>
        <div>
          <SimulationLabel>Path Steps: {pathSteps}</SimulationLabel>
          <Slider value={[pathSteps]} onValueChange={([v]) => setPathSteps(v)} min={50} max={1200} step={50} />
        </div>
      </SimulationConfig>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          Returned before cutoff: 1D {finite1d.length}/{trials} | 2D {finite2d.length}/{trials}
        </div>
      </SimulationResults>

      {/* p5.js animated 2D walk trail */}
      <SimulationMain
        scaleMode="contain"
        className="w-full rounded-lg overflow-hidden"
        style={{ background: isDark ? '#0a0a0f' : '#f0f4ff', aspectRatio: '1 / 1' }}
      >
        <WalkCanvas path={path} pathSteps={pathSteps} isDark={isDark} />
      </SimulationMain>

      {/* Histograms */}
      <SimulationAux>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <CanvasChart
            data={[{ x: finite1d, type: 'histogram', marker: { color: '#60a5fa' }, nbinsx: 40 }]}
            layout={{
              title: { text: 'First Return Distribution (1D)', font: { size: 13 } },
              xaxis: { title: { text: 'Return Step' } },
              yaxis: { title: { text: 'Count' } },
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            style={{ width: '100%', height: 320 }}
          />
          <CanvasChart
            data={[{ x: finite2d, type: 'histogram', marker: { color: '#34d399' }, nbinsx: 40 }]}
            layout={{
              title: { text: 'First Return Distribution (2D)', font: { size: 13 } },
              xaxis: { title: { text: 'Return Step' } },
              yaxis: { title: { text: 'Count' } },
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            style={{ width: '100%', height: 320 }}
          />
        </div>
      </SimulationAux>
    </SimulationPanel>
  );
}
