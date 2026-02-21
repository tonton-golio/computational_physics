'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { useTheme } from '@/lib/use-theme';

interface BakSneppenResult {
  chains: number[][];
  meanValues: number[];
  idxArr: number[];
  skipInit: number;
  avalancheTSpans: number[];
  avalancheXSpans: number[];
}

function runBakSneppen(size: number, nsteps: number): BakSneppenResult {
  const chain = Array.from({ length: size }, () => Math.random());
  const chains: number[][] = [];
  const idxArr: number[] = [];

  for (let n = 0; n < nsteps; n++) {
    let minIdx = 0;
    let minVal = chain[0];
    for (let i = 1; i < size; i++) {
      if (chain[i] < minVal) {
        minVal = chain[i];
        minIdx = i;
      }
    }
    idxArr.push(minIdx);

    // Replace min and its neighbors with new random values
    for (const offset of [-1, 0, 1]) {
      const idx = ((minIdx + offset) % size + size) % size;
      chain[idx] = Math.random();
    }
    chains.push([...chain]);
  }

  // Calculate mean values over time
  const meanValues = chains.map(c => c.reduce((s, v) => s + v, 0) / c.length);

  // Find skip-init point (when mean stabilizes)
  const patience = Math.min(500, Math.floor(nsteps / 4));
  let skipInit = patience;
  for (let i = patience; i < meanValues.length; i++) {
    const windowMean = meanValues.slice(i - patience, i).reduce((s, v) => s + v, 0) / patience;
    if (Math.abs(meanValues[i] - windowMean) < 0.001) {
      skipInit = i;
      break;
    }
  }

  // Compute avalanches from steady state
  const steadyIdx = idxArr.slice(skipInit);
  const avalancheTSpans: number[] = [];
  const avalancheXSpans: number[] = [];
  let counter = 0;
  const indices: number[] = [];

  for (let i = 1; i < steadyIdx.length; i++) {
    const isAvalanche = Math.abs(steadyIdx[i] - steadyIdx[i - 1]) < 2;
    if (isAvalanche) {
      counter++;
      indices.push(steadyIdx[i]);
    } else if (counter > 0) {
      avalancheTSpans.push(counter);
      avalancheXSpans.push(Math.max(...indices) - Math.min(...indices) + 1);
      counter = 0;
      indices.length = 0;
    }
  }

  return { chains, meanValues, idxArr, skipInit, avalancheTSpans, avalancheXSpans };
}

// ── p5.js Evolution Heatmap Canvas ─────────────────────────────────────

function EvolutionCanvas({ chains, size, isDark }: { chains: number[][]; size: number; isDark: boolean }) {
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
        const canvasWidth = containerRef.current?.clientWidth || 800;
        const canvasHeight = 350;
        const rows = chains.length;
        const cols = size;

        p.setup = () => {
          p.createCanvas(canvasWidth, canvasHeight);
          p.pixelDensity(1);
          p.noLoop();
          render(p, canvasWidth, canvasHeight);
        };

        function valueToColor(v: number): [number, number, number] {
          const t = Math.max(0, Math.min(1, v));

          const stops: { t: number; r: number; g: number; b: number }[] = [
            { t: 0.0,  r: 0x00, g: 0x33, b: 0xff },
            { t: 0.25, r: 0x00, g: 0xcc, b: 0xff },
            { t: 0.5,  r: 0xcc, g: 0xff, b: 0x00 },
            { t: 0.75, r: 0xff, g: 0x88, b: 0x00 },
            { t: 1.0,  r: 0xff, g: 0x00, b: 0x22 },
          ];

          let lo = stops[0];
          let hi = stops[stops.length - 1];
          for (let i = 0; i < stops.length - 1; i++) {
            if (t >= stops[i].t && t <= stops[i + 1].t) {
              lo = stops[i];
              hi = stops[i + 1];
              break;
            }
          }

          const range = hi.t - lo.t;
          const f = range === 0 ? 0 : (t - lo.t) / range;

          return [
            Math.round(lo.r + (hi.r - lo.r) * f),
            Math.round(lo.g + (hi.g - lo.g) * f),
            Math.round(lo.b + (hi.b - lo.b) * f),
          ];
        }

        function render(p: any, w: number, h: number) {
          const base = p.createGraphics(w, h);
          base.pixelDensity(1);
          base.loadPixels();

          for (let py = 0; py < h; py++) {
            const rowIdx = Math.floor((py / h) * rows);
            const row = chains[Math.min(rowIdx, rows - 1)];
            for (let px = 0; px < w; px++) {
              const colIdx = Math.floor((px / w) * cols);
              const val = row[Math.min(colIdx, cols - 1)];
              const [r, g, b] = valueToColor(val);
              const idx = (py * w + px) * 4;
              base.pixels[idx] = r;
              base.pixels[idx + 1] = g;
              base.pixels[idx + 2] = b;
              base.pixels[idx + 3] = 255;
            }
          }
          base.updatePixels();

          // Composite with glow via native canvas blur
          if (isDark) {
            p.background(10, 10, 15);
          } else {
            p.background(240, 244, 255);
          }
          const ctx = p.drawingContext as CanvasRenderingContext2D;
          if (isDark) {
            ctx.save();
            ctx.filter = 'blur(3px)';
            ctx.globalAlpha = 40 / 255;
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
  }, [chains, size, isDark]);

  return <div ref={containerRef} className="w-full" />;
}

// ── Main Component ─────────────────────────────────────────────────────

export function BakSneppen() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [size, setSize] = useState(200);
  const [nsteps, setNsteps] = useState(8000);
  const [seed, setSeed] = useState(0);

  const result = useMemo(() => {
    return runBakSneppen(size, nsteps);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, nsteps, seed]);

  // Subsample the chains for the heatmap display (take every Nth row)
  const subsampleRate = Math.max(1, Math.floor(result.chains.length / 300));
  const subsampledChains = result.chains.filter((_, i) => i % subsampleRate === 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Chain Size: {size}</label>
          <Slider
            min={50}
            max={500}
            step={10}
            value={[size]}
            onValueChange={([v]) => setSize(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Steps: {nsteps}</label>
          <Slider
            min={1000}
            max={30000}
            step={500}
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

      {/* Chain evolution heatmap (p5.js) */}
      <div>
        <div className="text-xs text-[var(--text-muted)] mb-1 font-medium">
          Bak-Sneppen Evolution (chain values over time)
        </div>
        <div className="rounded-lg overflow-hidden" style={{ background: isDark ? '#0a0a0f' : '#f0f4ff' }}>
          <EvolutionCanvas chains={subsampledChains} size={size} isDark={isDark} />
        </div>
        <div className="flex justify-between text-xs text-[var(--text-muted)] mt-1 px-1">
          <span>Site index 0</span>
          <span>Site index {size - 1}</span>
        </div>
      </div>

      {/* Mean value and argmin plots */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CanvasChart
          data={[
            {
              y: result.meanValues,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#3b82f6', width: 1 },
              name: 'Mean value',
            },
            {
              x: [result.skipInit, result.skipInit],
              y: [Math.min(...result.meanValues), Math.max(...result.meanValues)],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#ef4444', width: 2, dash: 'dash' },
              name: 'Steady state',
            },
          ]}
          layout={{
            title: { text: 'Average Value Over Time', font: { size: 13 } },
            xaxis: { title: { text: 'Timestep' } },
            yaxis: { title: { text: 'Mean' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          }}
          style={{ width: '100%', height: 280 }}
        />
        <CanvasChart
          data={[
            {
              y: result.idxArr,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#10b981', width: 1 },
              name: 'Argmin index',
            },
            {
              x: [result.skipInit, result.skipInit],
              y: [0, size],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#ef4444', width: 2, dash: 'dash' },
              name: 'Steady state',
            },
          ]}
          layout={{
            title: { text: 'Min-fitness Index Over Time', font: { size: 13 } },
            xaxis: { title: { text: 'Timestep' } },
            yaxis: { title: { text: 'Argmin' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          }}
          style={{ width: '100%', height: 280 }}
        />
      </div>

      {/* Avalanche distributions */}
      {result.avalancheTSpans.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <CanvasChart
            data={[{
              x: result.avalancheTSpans,
              type: 'histogram',
              marker: { color: '#8b5cf6' },
              nbinsx: 20,
            }]}
            layout={{
              title: { text: 'Avalanche Duration Distribution', font: { size: 13 } },
              xaxis: { title: { text: 'Duration (timesteps)' }, type: 'log' },
              yaxis: { title: { text: 'Frequency' }, type: 'log' },
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            style={{ width: '100%', height: 280 }}
          />
          <CanvasChart
            data={[{
              x: result.avalancheXSpans,
              type: 'histogram',
              marker: { color: '#ec4899' },
              nbinsx: 20,
            }]}
            layout={{
              title: { text: 'Avalanche Spatial Span Distribution', font: { size: 13 } },
              xaxis: { title: { text: 'Spatial span' }, type: 'log' },
              yaxis: { title: { text: 'Frequency' }, type: 'log' },
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            style={{ width: '100%', height: 280 }}
          />
        </div>
      )}
    </div>
  );
}
