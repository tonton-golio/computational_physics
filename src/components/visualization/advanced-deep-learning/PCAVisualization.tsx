"use client";

import { useState, useMemo } from 'react';
import { gaussianPair } from '@/lib/math';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationCheckbox } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function PCAVisualization({}: SimulationComponentProps) {
  const [nClusters, setNClusters] = useState(3);
  const [nPoints, setNPoints] = useState(80);
  const [spread, setSpread] = useState(0.6);
  const [showPCs, setShowPCs] = useState(true);
  const [showProjection, setShowProjection] = useState(false);

  const { points, pc1, pc2, eigenvalues, center, pc1Dir } = useMemo(() => {
    const rng = mulberry32(42);

    const clusterCenters: [number, number][] = [];
    for (let c = 0; c < nClusters; c++) {
      clusterCenters.push([
        gaussianPair(rng)[0] * 3,
        gaussianPair(rng)[0] * 3,
      ]);
    }

    const pts: { x: number; y: number; cluster: number }[] = [];
    const pointsPerCluster = Math.floor(nPoints / nClusters);
    for (let c = 0; c < nClusters; c++) {
      for (let i = 0; i < pointsPerCluster; i++) {
        pts.push({
          x: clusterCenters[c][0] + gaussianPair(rng)[0] * spread,
          y: clusterCenters[c][1] + gaussianPair(rng)[0] * spread,
          cluster: c,
        });
      }
    }

    const meanX = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const meanY = pts.reduce((s, p) => s + p.y, 0) / pts.length;

    let cxx = 0, cxy = 0, cyy = 0;
    for (const p of pts) {
      const dx = p.x - meanX;
      const dy = p.y - meanY;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    }
    const n = pts.length;
    cxx /= n;
    cxy /= n;
    cyy /= n;

    const trace = cxx + cyy;
    const det = cxx * cyy - cxy * cxy;
    const discriminant = Math.sqrt(Math.max(0, trace * trace / 4 - det));
    const lambda1 = trace / 2 + discriminant;
    const lambda2 = trace / 2 - discriminant;

    let v1x: number, v1y: number, v2x: number, v2y: number;
    if (Math.abs(cxy) > 1e-10) {
      v1x = lambda1 - cyy;
      v1y = cxy;
      v2x = lambda2 - cyy;
      v2y = cxy;
    } else {
      v1x = 1; v1y = 0; v2x = 0; v2y = 1;
    }

    const len1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const len2 = Math.sqrt(v2x * v2x + v2y * v2y);
    v1x /= len1; v1y /= len1;
    v2x /= len2; v2y /= len2;

    const scale1 = Math.sqrt(lambda1) * 2;
    const scale2 = Math.sqrt(lambda2) * 2;

    return {
      points: pts,
      pc1: { x: v1x * scale1, y: v1y * scale1 },
      pc2: { x: v2x * scale2, y: v2y * scale2 },
      eigenvalues: [lambda1, lambda2],
      center: { x: meanX, y: meanY },
      pc1Dir: { x: v1x, y: v1y },
    };
  }, [nClusters, nPoints, spread]);

  const clusterColors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7'];
  const totalVar = eigenvalues[0] + eigenvalues[1];

  const plotData: any[] = [];

  // Add scatter points for each cluster
  for (let c = 0; c < nClusters; c++) {
    const clusterPts = points.filter((p) => p.cluster === c);
    plotData.push({
      type: 'scatter' as const,
      x: clusterPts.map((p) => p.x),
      y: clusterPts.map((p) => p.y),
      mode: 'markers' as const,
      marker: {
        color: clusterColors[c % clusterColors.length],
        size: 7,
        opacity: 0.8,
      },
      name: `Cluster ${c + 1}`,
    });
  }

  // Add projected points onto PC1
  if (showProjection) {
    for (let c = 0; c < nClusters; c++) {
      const clusterPts = points.filter((p) => p.cluster === c);
      const projected = clusterPts.map(p => {
        const dx = p.x - center.x;
        const dy = p.y - center.y;
        const proj = dx * pc1Dir.x + dy * pc1Dir.y;
        return {
          x: center.x + proj * pc1Dir.x,
          y: center.y + proj * pc1Dir.y,
        };
      });
      // Lines from original to projected
      const lineX: (number | null)[] = [];
      const lineY: (number | null)[] = [];
      clusterPts.forEach((p, i) => {
        lineX.push(p.x, projected[i].x, null);
        lineY.push(p.y, projected[i].y, null);
      });
      plotData.push({
        type: 'scatter' as const,
        x: lineX,
        y: lineY,
        mode: 'lines' as const,
        line: { color: clusterColors[c % clusterColors.length], width: 0.8, dash: 'dot' },
        name: `Projection C${c + 1}`,
        showlegend: false,
        opacity: 0.4,
      });
      // Projected markers
      plotData.push({
        type: 'scatter' as const,
        x: projected.map(p => p.x),
        y: projected.map(p => p.y),
        mode: 'markers' as const,
        marker: {
          color: clusterColors[c % clusterColors.length],
          size: 5,
          symbol: 'x',
          opacity: 0.6,
        },
        name: `Proj. C${c + 1}`,
        showlegend: false,
      });
    }
  }

  // Add principal component arrows
  if (showPCs) {
    plotData.push({
      type: 'scatter' as const,
      x: [center.x - pc1.x, center.x, center.x + pc1.x],
      y: [center.y - pc1.y, center.y, center.y + pc1.y],
      mode: 'lines' as const,
      line: { color: '#f39c12', width: 3 },
      name: `PC1 (${(eigenvalues[0] / totalVar * 100).toFixed(1)}%)`,
    });
    plotData.push({
      type: 'scatter' as const,
      x: [center.x - pc2.x, center.x, center.x + pc2.x],
      y: [center.y - pc2.y, center.y, center.y + pc2.y],
      mode: 'lines' as const,
      line: { color: '#e74c3c', width: 3, dash: 'dash' },
      name: `PC2 (${(eigenvalues[1] / totalVar * 100).toFixed(1)}%)`,
    });
    plotData.push({
      type: 'scatter' as const,
      x: [center.x],
      y: [center.y],
      mode: 'markers' as const,
      marker: { color: '#ffffff', size: 12, symbol: 'x' },
      name: 'Mean',
      showlegend: false,
    });
  }

  return (
    <SimulationPanel title="Interactive PCA Visualization">
      <SimulationSettings>
        <SimulationCheckbox checked={showPCs} onChange={setShowPCs} label="Show Principal Components" />
        <SimulationCheckbox checked={showProjection} onChange={setShowProjection} label="Show Projection onto PC1" />
      </SimulationSettings>
      <SimulationConfig>
        <div className="space-y-4">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Number of Clusters: {nClusters}
            </SimulationLabel>
            <Slider min={2} max={5} step={1} value={[nClusters]} onValueChange={([v]) => setNClusters(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Points per Cluster: {Math.floor(nPoints / nClusters)}
            </SimulationLabel>
            <Slider min={30} max={200} step={10} value={[nPoints]} onValueChange={([v]) => setNPoints(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Cluster Spread: {spread.toFixed(2)}
            </SimulationLabel>
            <Slider min={0.1} max={2.0} step={0.1} value={[spread]} onValueChange={([v]) => setSpread(v)} className="w-full" />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
          data={plotData}
          layout={{
            xaxis: { title: { text: 'Feature 1' } },
            yaxis: { title: { text: 'Feature 2' }, scaleanchor: 'x' },
            margin: { t: 30, b: 50, l: 60, r: 30 },
            autosize: true,
          }}
          style={{ width: '100%', height: '450px' }}
        />
      </SimulationMain>
      <SimulationResults>
        <div className="p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
          <p className="font-semibold text-[var(--text-strong)] mb-2">Explained Variance:</p>
          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span style={{ color: '#f39c12' }}>PC1</span>
                <span>{totalVar > 0 ? ((eigenvalues[0] / totalVar) * 100).toFixed(1) : 0}%</span>
              </div>
              <div className="h-3 bg-[var(--surface-1)] rounded overflow-hidden">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${totalVar > 0 ? (eigenvalues[0] / totalVar) * 100 : 0}%`,
                    backgroundColor: '#f39c12',
                  }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span style={{ color: '#e74c3c' }}>PC2</span>
                <span>{totalVar > 0 ? ((eigenvalues[1] / totalVar) * 100).toFixed(1) : 0}%</span>
              </div>
              <div className="h-3 bg-[var(--surface-1)] rounded overflow-hidden">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${totalVar > 0 ? (eigenvalues[1] / totalVar) * 100 : 0}%`,
                    backgroundColor: '#e74c3c',
                  }}
                />
              </div>
            </div>
          </div>
          {showProjection && (
            <p className="mt-2 text-xs text-blue-400/70">
              PC1 captures {totalVar > 0 ? ((eigenvalues[0] / totalVar) * 100).toFixed(1) : 0}% of total variance because the data varies most along this direction. Projecting onto PC1 preserves this information while reducing to 1D.
            </p>
          )}
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
