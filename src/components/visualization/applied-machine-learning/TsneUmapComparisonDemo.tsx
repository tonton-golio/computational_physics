"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { gaussianNoise } from '@/lib/math';

export default function TsneUmapComparisonDemo({}: SimulationComponentProps): React.ReactElement {
  const [clusters, setClusters] = useState(3);
  const [spread, setSpread] = useState(0.5);
  const points = useMemo(() => {
    const centers = Array.from({ length: clusters }, (_, i) => {
      const angle = (2 * Math.PI * i) / clusters;
      return { x: 2 * Math.cos(angle), y: 2 * Math.sin(angle) };
    });
    const xs: number[] = [];
    const ys: number[] = [];
    const labels: number[] = [];
    for (let c = 0; c < centers.length; c++) {
      for (let i = 0; i < 120; i++) {
        const seedBase = c * 1000 + i * 7;
        xs.push(centers[c].x + gaussianNoise(seedBase + 1, spread));
        ys.push(centers[c].y + gaussianNoise(seedBase + 2, spread));
        labels.push(c);
      }
    }
    const pcaX = xs.map((v, i) => 0.8 * v + 0.2 * ys[i]);
    const pcaY = ys.map((v, i) => -0.2 * xs[i] + 0.8 * v);
    const tsneX = xs.map((v, i) => Math.tanh(0.7 * v) + 0.25 * Math.sin(ys[i]));
    const tsneY = ys.map((v, i) => Math.tanh(0.7 * v) + 0.25 * Math.cos(xs[i]));
    const umapX = xs.map((v, i) => 0.9 * Math.tanh(v) + 0.1 * ys[i]);
    const umapY = ys.map((v, i) => 0.9 * Math.tanh(v) - 0.1 * xs[i]);

    // Map cluster index to colors
    const clusterColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
    const pointColors = labels.map((l) => clusterColors[l % clusterColors.length]);
    return { pointColors, pcaX, pcaY, tsneX, tsneY, umapX, umapY };
  }, [clusters, spread]);

  const commonLayout = {
    margin: { t: 20, r: 20, b: 40, l: 40 },
    showlegend: false,
  };

  return (
    <SimulationPanel title="PCA vs t-SNE vs UMAP (Intuition)">
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <SimulationLabel>Cluster count: {clusters}</SimulationLabel>
            <Slider min={2} max={6} step={1} value={[clusters]} onValueChange={([v]) => setClusters(v)} />
          </div>
          <div>
            <SimulationLabel>Spread: {spread.toFixed(2)}</SimulationLabel>
            <Slider min={0.15} max={1.2} step={0.05} value={[spread]} onValueChange={([v]) => setSpread(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">PCA</p>
          <CanvasChart
            data={[
              {
                x: points.pcaX,
                y: points.pcaY,
                type: 'scatter',
                mode: 'markers',
                marker: { size: 5, color: points.pointColors },
                name: 'PCA',
              },
            ]}
            layout={commonLayout}
            style={{ width: '100%', height: 280 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">t-SNE</p>
          <CanvasChart
            data={[
              {
                x: points.tsneX,
                y: points.tsneY,
                type: 'scatter',
                mode: 'markers',
                marker: { size: 5, color: points.pointColors },
                name: 't-SNE',
              },
            ]}
            layout={commonLayout}
            style={{ width: '100%', height: 280 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">UMAP</p>
          <CanvasChart
            data={[
              {
                x: points.umapX,
                y: points.umapY,
                type: 'scatter',
                mode: 'markers',
                marker: { size: 5, color: points.pointColors },
                name: 'UMAP',
              },
            ]}
            layout={commonLayout}
            style={{ width: '100%', height: 280 }}
          />
        </div>
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
