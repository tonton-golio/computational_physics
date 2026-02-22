"use client";

import { useMemo, useState, useRef, useEffect } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationAux, SimulationLabel } from '@/components/ui/simulation-panel';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { useTheme } from '@/lib/use-theme';

function mandelbrotMask(resolution: number, maxIter: number, exponent: number): number[][] {
  const mask: number[][] = [];
  const xMin = -2.2;
  const xMax = 1.2;
  const yMin = -1.4;
  const yMax = 1.4;
  for (let i = 0; i < resolution; i++) {
    const row: number[] = [];
    const y = yMin + (i / (resolution - 1)) * (yMax - yMin);
    for (let j = 0; j < resolution; j++) {
      const x = xMin + (j / (resolution - 1)) * (xMax - xMin);
      let zr = 0;
      let zi = 0;
      let stable = 1;
      for (let n = 0; n < maxIter; n++) {
        const r = Math.sqrt(zr * zr + zi * zi);
        const theta = Math.atan2(zi, zr);
        const rPow = Math.pow(r, exponent);
        zr = rPow * Math.cos(exponent * theta) + x;
        zi = rPow * Math.sin(exponent * theta) + y;
        if (zr * zr + zi * zi > 4) {
          stable = 0;
          break;
        }
      }
      row.push(stable);
    }
    mask.push(row);
  }
  return mask;
}

function boxCountDimension(mask: number[][]): { eps: number[]; counts: number[]; slope: number } {
  const n = mask.length;
  const sizes = [2, 4, 8, 16, 32].filter(s => s <= n);
  const eps: number[] = [];
  const counts: number[] = [];
  for (const box of sizes) {
    let count = 0;
    const step = Math.floor(n / box);
    if (step < 1) continue;
    for (let i = 0; i < n; i += step) {
      for (let j = 0; j < n; j += step) {
        let has = false;
        for (let ii = i; ii < Math.min(i + step, n) && !has; ii++) {
          for (let jj = j; jj < Math.min(j + step, n); jj++) {
            if (mask[ii][jj] === 1) {
              has = true;
              break;
            }
          }
        }
        if (has) count++;
      }
    }
    eps.push(1 / box);
    counts.push(Math.max(count, 1));
  }

  const x = eps.map(v => Math.log(1 / v));
  const y = counts.map(v => Math.log(v));
  const nPts = x.length;
  let slope = 0;
  if (nPts >= 2) {
    const sx = x.reduce((s, v) => s + v, 0);
    const sy = y.reduce((s, v) => s + v, 0);
    const sxy = x.reduce((s, v, i) => s + v * y[i], 0);
    const sx2 = x.reduce((s, v) => s + v * v, 0);
    slope = (nPts * sxy - sx * sy) / (nPts * sx2 - sx * sx);
  }
  return { eps, counts, slope };
}

// ── p5.js Fractal Canvas ──────────────────────────────────────────────

function FractalCanvas({ mask, resolution, isDark }: { mask: number[][]; resolution: number; isDark: boolean }) {
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
        const canvasHeight = Math.round(canvasWidth * (2.8 / 3.4));

        p.setup = () => {
          p.createCanvas(canvasWidth, canvasHeight);
          p.pixelDensity(1);
          p.noLoop();
          renderFractal(p, canvasWidth, canvasHeight);
        };

        function renderFractal(p: any, w: number, h: number) {
          const base = p.createGraphics(w, h);
          base.pixelDensity(1);
          base.loadPixels();

          for (let py = 0; py < h; py++) {
            for (let px = 0; px < w; px++) {
              const mi = Math.floor((py / h) * resolution);
              const mj = Math.floor((px / w) * resolution);
              const val =
                mask[Math.min(mi, resolution - 1)]?.[Math.min(mj, resolution - 1)] ?? 0;

              const idx = (py * w + px) * 4;
              if (val === 1) {
                if (isDark) {
                  base.pixels[idx] = 0;
                  base.pixels[idx + 1] = 255;
                  base.pixels[idx + 2] = 221;
                } else {
                  base.pixels[idx] = 0;
                  base.pixels[idx + 1] = 160;
                  base.pixels[idx + 2] = 140;
                }
                base.pixels[idx + 3] = 255;
              } else {
                if (isDark) {
                  base.pixels[idx] = 11;
                  base.pixels[idx + 1] = 18;
                  base.pixels[idx + 2] = 32;
                } else {
                  base.pixels[idx] = 223;
                  base.pixels[idx + 1] = 232;
                  base.pixels[idx + 2] = 251;
                }
                base.pixels[idx + 3] = 255;
              }
            }
          }
          base.updatePixels();

          if (isDark) {
            p.background(11, 18, 32);
          } else {
            p.background(240, 244, 255);
          }
          const ctx = p.drawingContext as CanvasRenderingContext2D;

          if (isDark) {
            ctx.save();
            ctx.filter = 'blur(12px)';
            ctx.globalAlpha = 35 / 255;
            ctx.drawImage(base.elt, 0, 0);
            ctx.restore();

            ctx.save();
            ctx.filter = 'blur(6px)';
            ctx.globalAlpha = 55 / 255;
            ctx.drawImage(base.elt, 0, 0);
            ctx.restore();
          }

          p.image(base, 0, 0);

          base.remove();
        }
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
  }, [mask, resolution, isDark]);

  return <div ref={containerRef} className="w-full" />;
}

// ── Main Component ────────────────────────────────────────────────────

export default function FractalDimension({}: SimulationComponentProps) {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [resolution, setResolution] = useState(128);
  const [maxIter, setMaxIter] = useState(35);
  const [exponent, setExponent] = useState(2.0);

  const { mask, boxes } = useMemo(() => {
    const m = mandelbrotMask(resolution, maxIter, exponent);
    return { mask: m, boxes: boxCountDimension(m) };
  }, [resolution, maxIter, exponent]);

  return (
    <SimulationPanel title="Fractal Dimension (Box Counting)">
      <SimulationConfig>
        <div>
          <SimulationLabel>Resolution: {resolution}</SimulationLabel>
          <Slider value={[resolution]} onValueChange={([v]) => setResolution(v)} min={64} max={512} step={64} />
        </div>
        <div>
          <SimulationLabel>Iterations: {maxIter}</SimulationLabel>
          <Slider value={[maxIter]} onValueChange={([v]) => setMaxIter(v)} min={10} max={80} step={1} />
        </div>
        <div>
          <SimulationLabel>Exponent a: {exponent.toFixed(1)}</SimulationLabel>
          <Slider value={[exponent]} onValueChange={([v]) => setExponent(v)} min={1.5} max={4.0} step={0.1} />
        </div>
      </SimulationConfig>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">Estimated box-counting fractal dimension D ≈ {boxes.slope.toFixed(3)}</div>
      </SimulationResults>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* p5.js Fractal Rendering */}
        <SimulationMain scaleMode="contain" className="rounded-lg overflow-hidden" style={{ background: isDark ? '#0a0a0f' : '#f0f4ff' }}>
          <FractalCanvas mask={mask} resolution={resolution} isDark={isDark} />
        </SimulationMain>

        {/* Box-counting log-log plot */}
        <SimulationAux>
          <CanvasChart
            data={[
              {
                x: boxes.eps.map(e => Math.log(1 / e)),
                y: boxes.counts.map(v => Math.log(v)),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#f59e0b', width: 2 },
                marker: { size: 6 },
              },
            ]}
            layout={{
              title: { text: 'Box Counting: log(N) vs log(1/epsilon)', font: { size: 13 } },
              xaxis: { title: { text: 'log(1/epsilon)' } },
              yaxis: { title: { text: 'log(N)' } },
              showlegend: false,
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            style={{ width: '100%', height: 360 }}
          />
        </SimulationAux>
      </div>
    </SimulationPanel>
  );
}
