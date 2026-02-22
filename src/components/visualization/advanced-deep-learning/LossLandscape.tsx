"use client";

import { useState, useMemo } from 'react';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { Slider } from '@/components/ui/slider';
import { useTheme } from '@/lib/use-theme';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function computeLandscape(sharpness: number, numMinima: number, resolution: number) {
  const range = 4;
  const step = (2 * range) / (resolution - 1);
  const x: number[] = [];
  const y: number[] = [];
  const z: number[][] = [];

  for (let i = 0; i < resolution; i++) {
    const yi = -range + i * step;
    y.push(yi);
    const row: number[] = [];
    for (let j = 0; j < resolution; j++) {
      const xi = -range + j * step;
      if (i === 0) x.push(xi);

      // Base quadratic bowl
      let val = 0.1 * (xi * xi + yi * yi);

      // Add local minima
      if (numMinima >= 2) {
        val -= 1.5 * Math.exp(-sharpness * ((xi - 1.5) ** 2 + (yi - 1) ** 2));
        val -= 1.2 * Math.exp(-sharpness * ((xi + 1.5) ** 2 + (yi + 0.8) ** 2));
      }
      if (numMinima >= 3) {
        val -= 0.8 * Math.exp(-sharpness * 0.6 * ((xi - 0.5) ** 2 + (yi + 2) ** 2));
      }

      // Saddle point contribution
      val += 0.3 * Math.exp(-0.5 * (xi * xi + yi * yi)) * (xi * xi - yi * yi) * 0.1;

      row.push(val);
    }
    z.push(row);
  }

  // Find critical points
  const criticalPoints: { x: number; y: number; z: number; type: string }[] = [];
  // Global minimum approximation
  let minVal = Infinity;
  let minX = 0, minY = 0;
  for (let i = 1; i < resolution - 1; i++) {
    for (let j = 1; j < resolution - 1; j++) {
      if (z[i][j] < minVal) {
        minVal = z[i][j];
        minX = x[j];
        minY = y[i];
      }
    }
  }
  criticalPoints.push({ x: minX, y: minY, z: minVal, type: 'Global minimum' });

  return { x, y, z, criticalPoints };
}

export default function LossLandscape({}: SimulationComponentProps) {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [sharpness, setSharpness] = useState(1.5);
  const [numMinima, setNumMinima] = useState(2);
  const [viewAngle, setViewAngle] = useState(30);

  const resolution = 60;
  const data = useMemo(() => computeLandscape(sharpness, numMinima, resolution), [sharpness, numMinima]);

  // Surface rendered as 2D heatmap (top-down view of loss landscape)
  const heatmapTrace: any = {
    type: 'heatmap' as const,
    x: data.x,
    y: data.y,
    z: data.z,
    colorscale: isDark
      ? [
          [0, '#0d0d2b'],
          [0.2, '#1a1a5e'],
          [0.4, '#3b2b8e'],
          [0.6, '#6b3fa0'],
          [0.8, '#c05299'],
          [1, '#f06292'],
        ]
      : [
          [0, '#e8edf8'],
          [0.2, '#c5b3e8'],
          [0.4, '#9b6bc4'],
          [0.6, '#8b4fb0'],
          [0.8, '#c05299'],
          [1, '#f06292'],
        ],
    showscale: true,
  };

  // Overlay critical points as scatter markers
  const markerTrace: any = {
    type: 'scatter' as const,
    x: data.criticalPoints.map(p => p.x),
    y: data.criticalPoints.map(p => p.y),
    mode: 'markers+text' as const,
    marker: { size: 10, color: '#fbbf24', symbol: 'diamond' },
    text: data.criticalPoints.map(p => p.type),
    textposition: 'top center',
    textfont: { color: '#fbbf24', size: 10 },
    name: 'Critical points',
  };

  return (
    <SimulationPanel title="Loss Landscape Visualization">
      <SimulationConfig>
        <div className="space-y-4">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Minima sharpness: {sharpness.toFixed(1)}
            </SimulationLabel>
            <Slider min={0.3} max={3} step={0.1} value={[sharpness]} onValueChange={([v]) => setSharpness(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Number of minima: {numMinima}
            </SimulationLabel>
            <Slider min={2} max={3} step={1} value={[numMinima]} onValueChange={([v]) => setNumMinima(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Camera elevation: {viewAngle}
            </SimulationLabel>
            <Slider min={10} max={80} step={5} value={[viewAngle]} onValueChange={([v]) => setViewAngle(v)} className="w-full" />
          </div>

          <div className="p-3 bg-[var(--surface-2)] rounded text-xs text-[var(--text-muted)] space-y-2">
            <p><strong className="text-[var(--text-strong)]">Sharp minima</strong> (high curvature) tend to generalize poorly: small perturbations in parameters cause large loss changes.</p>
            <p><strong className="text-[var(--text-strong)]">Flat minima</strong> (low curvature) tend to generalize better: the solution is robust to perturbations.</p>
            <p>Saddle points (where curvature is positive in some directions and negative in others) are more common than local minima in high dimensions.</p>
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasHeatmap
          data={[heatmapTrace, markerTrace]}
          layout={{
            xaxis: { title: { text: 'w1' } },
            yaxis: { title: { text: 'w2' } },
            margin: { t: 20, b: 50, l: 60, r: 20 },
            autosize: true,
          }}
          style={{ width: '100%', height: '500px' }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
