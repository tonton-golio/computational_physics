"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

type Geometry = 'cross-borehole' | 'surface-to-surface' | 'combined';

export default function RaySmearingExplorer({}: SimulationComponentProps) {
  const [gridN, setGridN] = useState(10);
  const [geometry, setGeometry] = useState<Geometry>('cross-borehole');

  const result = useMemo(() => {
    const N = gridN;
    const hitCount: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    const rays: { x: number[]; y: number[] }[] = [];

    // Cross-borehole: sources on left, receivers on right, diagonal rays
    if (geometry === 'cross-borehole' || geometry === 'combined') {
      const nSrc = Math.max(3, Math.floor(N * 0.7));
      const nRec = Math.max(3, Math.floor(N * 0.7));
      for (let si = 0; si < nSrc; si++) {
        const sy = (si + 0.5) * N / nSrc;
        for (let ri = 0; ri < nRec; ri++) {
          const ry = (ri + 0.5) * N / nRec;
          const rx: number[] = [];
          const ryArr: number[] = [];
          const nSteps = N * 3;
          for (let s = 0; s <= nSteps; s++) {
            const t = s / nSteps;
            const px = t * N;
            const py = sy + t * (ry - sy);
            rx.push(px);
            ryArr.push(py);
            const ci = Math.min(Math.floor(px), N - 1);
            const cj = Math.min(Math.floor(py), N - 1);
            if (ci >= 0 && ci < N && cj >= 0 && cj < N) hitCount[cj][ci]++;
          }
          rays.push({ x: rx, y: ryArr });
        }
      }
    }

    // Surface-to-surface: sources and receivers on top, rays curving down
    if (geometry === 'surface-to-surface' || geometry === 'combined') {
      const nSrc = Math.max(3, Math.floor(N * 0.5));
      const nRec = Math.max(3, Math.floor(N * 0.5));
      for (let si = 0; si < nSrc; si++) {
        const sx = (si + 0.5) * N / nSrc;
        for (let ri = 0; ri < nRec; ri++) {
          const recX = (ri + 0.5) * N / nRec;
          if (Math.abs(sx - recX) < N / (nSrc * 2)) continue;
          const rx: number[] = [];
          const ryArr: number[] = [];
          const nSteps = N * 3;
          const depth = Math.abs(recX - sx) * 0.4;
          for (let s = 0; s <= nSteps; s++) {
            const t = s / nSteps;
            const px = sx + t * (recX - sx);
            const py = 4 * depth * t * (1 - t);
            rx.push(px);
            ryArr.push(py);
            const ci = Math.min(Math.floor(px), N - 1);
            const cj = Math.min(Math.floor(py), N - 1);
            if (ci >= 0 && ci < N && cj >= 0 && cj < N) hitCount[cj][ci]++;
          }
          rays.push({ x: rx, y: ryArr });
        }
      }
    }

    // Normalize hit counts for heatmap coloring
    let maxHit = 0;
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++)
        if (hitCount[r][c] > maxHit) maxHit = hitCount[r][c];

    // Build heatmap as scatter points with color
    const heatX: number[] = [];
    const heatY: number[] = [];
    const heatColor: string[] = [];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        heatX.push(c + 0.5);
        heatY.push(r + 0.5);
        const t = maxHit > 0 ? hitCount[r][c] / maxHit : 0;
        const red = Math.round(255 * t);
        const green = Math.round(100 * t);
        const blue = Math.round(255 * (1 - t) * 0.4);
        heatColor.push(`rgb(${red},${green},${blue})`);
      }
    }

    // Subsample rays for display (max 40 to keep chart readable)
    const stride = Math.max(1, Math.floor(rays.length / 40));
    const displayRays = rays.filter((_, i) => i % stride === 0).slice(0, 40);

    return { heatX, heatY, heatColor, displayRays, N, maxHit, totalRays: rays.length };
  }, [gridN, geometry]);

  return (
    <SimulationPanel title="Ray Smearing Explorer">
      <SimulationSettings>
        <SimulationToggle
          options={(['cross-borehole', 'surface-to-surface', 'combined'] as Geometry[]).map(g => ({
            label: g.replace(/-/g, ' '),
            value: g,
          }))}
          value={geometry}
          onChange={(v) => setGeometry(v as Geometry)}
          className="mt-1"
        />
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel className="mb-1 block text-sm text-[var(--text-muted)]">Grid size: {gridN}x{gridN}</SimulationLabel>
          <Slider value={[gridN]} onValueChange={([v]) => setGridN(v)} min={5} max={16} step={1} />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <CanvasChart
          data={[
            ...result.displayRays.map((ray, i) => ({
              x: ray.x, y: ray.y,
              type: 'scatter' as const, mode: 'lines' as const,
              line: { color: 'rgba(59,130,246,0.25)', width: 1 },
              showlegend: i === 0,
              name: i === 0 ? `Rays (${result.totalRays} total)` : '',
            })),
          ]}
          layout={{
            title: { text: 'Ray Paths' },
            xaxis: { title: { text: 'x cell' }, range: [0, result.N] },
            yaxis: { title: { text: 'depth cell' }, range: [result.N, 0] },
            height: 380,
            showlegend: true,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />

        <CanvasChart
          data={[{
            x: result.heatX, y: result.heatY,
            type: 'scatter', mode: 'markers',
            marker: {
              size: Math.max(4, Math.min(20, 200 / result.N)),
              color: result.heatColor,
              symbol: 'square',
            },
            name: 'Hit count',
            showlegend: false,
          }]}
          layout={{
            title: { text: `Ray Density Heatmap (max=${result.maxHit})` },
            xaxis: { title: { text: 'x cell' }, range: [0, result.N] },
            yaxis: { title: { text: 'depth cell' }, range: [result.N, 0] },
            height: 380,
            margin: { t: 40, b: 50, l: 50, r: 20 },
          }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          Hot (red) cells are well-sampled by many rays and will be well-resolved in tomographic
          inversion. Cold (dark) cells have poor coverage and cannot be reliably reconstructed.
          Switch between geometries to see how source-receiver layout controls resolution.
          Combined geometry fills in blind spots that individual geometries miss.
        </p>
      </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
