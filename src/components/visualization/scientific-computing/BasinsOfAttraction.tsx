'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Basins of attraction for Newton's method on z^3 - 1 = 0 in the complex plane.
 * Each pixel is colored by which of the three cube roots of unity Newton's method
 * converges to from that starting point. Fractal boundaries emerge.
 */

const POLYNOMIALS = [
  {
    label: 'z^3 - 1',
    roots: [
      [1, 0],
      [-0.5, Math.sqrt(3) / 2],
      [-0.5, -Math.sqrt(3) / 2],
    ] as [number, number][],
    f: (re: number, im: number): [number, number] => {
      // z^3 - 1
      const r2 = re * re - im * im;
      const i2 = 2 * re * im;
      return [re * r2 - im * i2 - 1, re * i2 + im * r2];
    },
    fp: (re: number, im: number): [number, number] => {
      // 3z^2
      return [3 * (re * re - im * im), 3 * 2 * re * im];
    },
  },
  {
    label: 'z^4 - 1',
    roots: [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
    ] as [number, number][],
    f: (re: number, im: number): [number, number] => {
      const r2 = re * re - im * im;
      const i2 = 2 * re * im;
      return [r2 * r2 - i2 * i2 - 1, 2 * r2 * i2];
    },
    fp: (re: number, im: number): [number, number] => {
      // 4z^3
      const r2 = re * re - im * im;
      const i2 = 2 * re * im;
      return [4 * (re * r2 - im * i2), 4 * (re * i2 + im * r2)];
    },
  },
  {
    label: 'z^5 - 1',
    roots: Array.from({ length: 5 }, (_, k) => [
      Math.cos((2 * Math.PI * k) / 5),
      Math.sin((2 * Math.PI * k) / 5),
    ]) as [number, number][],
    f: (re: number, im: number): [number, number] => {
      // z^5 - 1 via repeated multiplication
      let zr = re, zi = im;
      for (let i = 1; i < 5; i++) {
        const nr = zr * re - zi * im;
        const ni = zr * im + zi * re;
        zr = nr; zi = ni;
      }
      return [zr - 1, zi];
    },
    fp: (re: number, im: number): [number, number] => {
      // 5z^4
      let zr = re, zi = im;
      for (let i = 1; i < 4; i++) {
        const nr = zr * re - zi * im;
        const ni = zr * im + zi * re;
        zr = nr; zi = ni;
      }
      return [5 * zr, 5 * zi];
    },
  },
];

// Root colors (index-based)
const ROOT_HUES = [0, 0.33, 0.66, 0.17, 0.83];

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
  };
  return [f(0), f(8), f(4)];
}

export default function BasinsOfAttraction({}: SimulationComponentProps) {
  const [polyIdx, setPolyIdx] = useState(0);
  const [resolution, setResolution] = useState(200);
  const [maxIter, setMaxIter] = useState(30);
  const [zoom, setZoom] = useState(2.0);

  const poly = POLYNOMIALS[polyIdx];

  const basinData = useMemo(() => {
    const n = resolution;
    const z: number[][] = [];
    const xGrid: number[] = [];
    const yGrid: number[] = [];

    for (let i = 0; i < n; i++) {
      xGrid.push(-zoom + (2 * zoom * i) / (n - 1));
      yGrid.push(-zoom + (2 * zoom * i) / (n - 1));
    }

    for (let iy = 0; iy < n; iy++) {
      const row: number[] = [];
      for (let ix = 0; ix < n; ix++) {
        let re = xGrid[ix];
        let im = yGrid[iy];

        let rootIdx = -1;
        let convergedIter = maxIter;

        for (let k = 0; k < maxIter; k++) {
          const [fr, fi] = poly.f(re, im);
          const [dr, di] = poly.fp(re, im);

          const denom = dr * dr + di * di;
          if (denom < 1e-20) break;

          // Newton step: z - f(z)/f'(z)
          re -= (fr * dr + fi * di) / denom;
          im -= (fi * dr - fr * di) / denom;

          // Check convergence to each root
          for (let r = 0; r < poly.roots.length; r++) {
            const dx = re - poly.roots[r][0];
            const dy = im - poly.roots[r][1];
            if (dx * dx + dy * dy < 1e-6) {
              rootIdx = r;
              convergedIter = k;
              break;
            }
          }
          if (rootIdx >= 0) break;
        }

        // Encode: rootIdx * 100 + iteration count for shading
        if (rootIdx >= 0) {
          // Value encodes root and speed: low values = fast convergence
          row.push(rootIdx + (convergedIter / maxIter) * 0.9);
        } else {
          row.push(-1);
        }
      }
      z.push(row);
    }

    // Convert to RGB colorscale using custom heatmap values
    // We need to map to a 0-1 range for the heatmap
    const nRoots = poly.roots.length;
    const zNorm: number[][] = [];
    for (const row of z) {
      const normRow: number[] = [];
      for (const val of row) {
        if (val < 0) {
          normRow.push(0); // black for non-convergence
        } else {
          const rootIdx = Math.floor(val);
          const frac = val - rootIdx;
          // Map each root to a distinct range
          const base = rootIdx / nRoots;
          const darkness = (1 - frac * 0.7); // faster convergence = brighter
          normRow.push(base + darkness / nRoots * 0.8);
        }
      }
      zNorm.push(normRow);
    }

    return { z: zNorm, xGrid, yGrid };
  }, [poly, resolution, maxIter, zoom]);

  // Build a custom colorscale that assigns distinct hues to each root
  const colorscale = useMemo(() => {
    const nRoots = poly.roots.length;
    const stops: Array<[number, string]> = [];
    stops.push([0, 'rgb(10,10,10)']);
    for (let r = 0; r < nRoots; r++) {
      const base = r / nRoots;
      const end = (r + 1) / nRoots;
      const [cr, cg, cb] = hslToRgb(ROOT_HUES[r % ROOT_HUES.length], 0.85, 0.2);
      const [br, bg, bb] = hslToRgb(ROOT_HUES[r % ROOT_HUES.length], 0.9, 0.6);
      stops.push([base + 0.001, `rgb(${Math.round(cr * 255)},${Math.round(cg * 255)},${Math.round(cb * 255)})`]);
      stops.push([end - 0.001, `rgb(${Math.round(br * 255)},${Math.round(bg * 255)},${Math.round(bb * 255)})`]);
    }
    stops.push([1, `rgb(200,200,100)`]);
    return stops;
  }, [poly.roots.length]);

  return (
    <SimulationPanel>
      <h3 className="text-lg font-semibold text-[var(--text-strong)]">
        Newton&apos;s Method: Basins of Attraction
      </h3>
      <p className="text-sm text-[var(--text-soft)] mb-4">
        Each point in the complex plane is colored by which root Newton&apos;s method converges
        to. The fractal boundaries reveal the chaotic sensitivity to initial conditions.
        Brighter colors indicate faster convergence.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div>
          <SimulationLabel>Polynomial</SimulationLabel>
          <SimulationToggle
            options={POLYNOMIALS.map((p, i) => ({ label: p.label, value: String(i) }))}
            value={String(polyIdx)}
            onChange={(v) => setPolyIdx(Number(v))}
          />
        </div>
        <div>
          <SimulationLabel>Resolution: {resolution} x {resolution}</SimulationLabel>
          <Slider
            value={[resolution]}
            onValueChange={([v]) => setResolution(v)}
            min={80}
            max={400}
            step={20}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>Max iterations: {maxIter}</SimulationLabel>
          <Slider
            value={[maxIter]}
            onValueChange={([v]) => setMaxIter(v)}
            min={10}
            max={80}
            step={5}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>Zoom: {zoom.toFixed(1)}</SimulationLabel>
          <Slider
            value={[zoom]}
            onValueChange={([v]) => setZoom(v)}
            min={0.5}
            max={4}
            step={0.1}
            className="w-full"
          />
        </div>
      </div>

      <CanvasHeatmap
        data={[
          {
            z: basinData.z,
            x: basinData.xGrid,
            y: basinData.yGrid,
            colorscale,
            showscale: false,
          },
        ]}
        layout={{
          title: { text: `Basins for ${poly.label} = 0` },
          xaxis: { title: { text: 'Re(z)' } },
          yaxis: { title: { text: 'Im(z)' } },
          margin: { t: 40, r: 20, b: 50, l: 60 },
        }}
        style={{ width: '100%', height: 450 }}
      />

      <div className="mt-3 flex flex-wrap gap-3">
        {poly.roots.map((root, i) => {
          const [r, g, b] = hslToRgb(ROOT_HUES[i % ROOT_HUES.length], 0.9, 0.5);
          const color = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
          return (
            <div key={i} className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              Root {i + 1}: ({root[0].toFixed(3)}, {root[1].toFixed(3)}i)
            </div>
          );
        })}
      </div>
    </SimulationPanel>
  );
}
