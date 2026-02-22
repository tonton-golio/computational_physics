"use client";

import React, { useEffect, useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { generateSwissRoll, pca2d, tsne } from './ml-utils';

function normalizeRange(arr: number[]): number[] {
  let min = Infinity, max = -Infinity;
  for (const v of arr) { if (v < min) min = v; if (v > max) max = v; }
  const range = max - min || 1;
  return arr.map((v) => (v - min) / range);
}

function labelToColor(labels: number[]): string[] {
  const norm = normalizeRange(labels);
  return norm.map((t) => {
    const h = Math.round(t * 270); // violet-to-red through hue
    return `hsl(${h}, 85%, 55%)`;
  });
}

export default function SwissRollExplorer({}: SimulationComponentProps): React.ReactElement {
  const [nPoints, setNPoints] = useState(800);
  const [perplexity, setPerplexity] = useState(30);
  const [tsneIter, setTsneIter] = useState(300);
  const [computing, setComputing] = useState(false);

  // Generate Swiss Roll data
  const { data, labels } = useMemo(() => generateSwissRoll(nPoints, 42), [nPoints]);
  const colors = useMemo(() => labelToColor(labels), [labels]);

  // PCA projection (instant)
  const pcaResult = useMemo(() => pca2d(data), [data]);

  // t-SNE projection (computed async to avoid blocking UI)
  const [tsneResult, setTsneResult] = useState<number[][] | null>(null);

  // UMAP projection
  const [umapResult, setUmapResult] = useState<number[][] | null>(null);
  const [nNeighbors, setNNeighbors] = useState(15);

  // Run t-SNE when parameters change
  useEffect(() => {
    setComputing(true);
    setTsneResult(null);

    const timeout = setTimeout(() => {
      const result = tsne(data, { perplexity, iterations: tsneIter, seed: 42 });
      setTsneResult(result);
      setComputing(false);
    }, 50);

    return () => clearTimeout(timeout);
  }, [data, perplexity, tsneIter]);

  // Run UMAP when parameters change
  useEffect(() => {
    setUmapResult(null);
    let cancelled = false;

    const timeout = setTimeout(async () => {
      try {
        const { UMAP } = await import('umap-js');
        const umap = new UMAP({
          nNeighbors,
          nComponents: 2,
          minDist: 0.1,
          nEpochs: 200,
        });
        const embedding = umap.fit(data);
        if (!cancelled) {
          setUmapResult(embedding as number[][]);
        }
      } catch {
        // UMAP unavailable – show placeholder
        if (!cancelled) setUmapResult(null);
      }
    }, 50);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [data, nNeighbors]);

  const commonLayout = {
    margin: { t: 20, r: 20, b: 40, l: 40 },
    showlegend: false,
  };

  const makeTrace = (coords: number[][]) => ({
    x: coords.map((p) => p[0]),
    y: coords.map((p) => p[1]),
    type: 'scatter' as const,
    mode: 'markers' as const,
    marker: { size: 4, color: colors, opacity: 0.7 },
    name: '',
  });

  const loadingPlaceholder = (
    <div className="flex h-[280px] items-center justify-center text-sm text-[var(--text-muted)]">
      Computing...
    </div>
  );

  return (
    <SimulationPanel title="Swiss Roll Manifold Explorer">
      <SimulationConfig>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <div>
            <SimulationLabel>Points: {nPoints}</SimulationLabel>
            <Slider min={200} max={2000} step={100} value={[nPoints]} onValueChange={([v]) => setNPoints(v)} />
          </div>
          <div>
            <SimulationLabel>Perplexity (t-SNE): {perplexity}</SimulationLabel>
            <Slider min={5} max={100} step={5} value={[perplexity]} onValueChange={([v]) => setPerplexity(v)} />
          </div>
          <div>
            <SimulationLabel>n_neighbors (UMAP): {nNeighbors}</SimulationLabel>
            <Slider min={5} max={50} step={5} value={[nNeighbors]} onValueChange={([v]) => setNNeighbors(v)} />
          </div>
          <div>
            <SimulationLabel>t-SNE iterations: {tsneIter}</SimulationLabel>
            <Slider min={100} max={800} step={50} value={[tsneIter]} onValueChange={([v]) => setTsneIter(v)} />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <p className="mb-1 text-center text-sm font-medium text-[var(--text-muted)]">PCA</p>
          <CanvasChart
            data={[makeTrace(pcaResult)]}
            layout={commonLayout}
            style={{ width: '100%', height: 280 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium text-[var(--text-muted)]">
            t-SNE {computing && '(computing...)'}
          </p>
          {tsneResult ? (
            <CanvasChart
              data={[makeTrace(tsneResult)]}
              layout={commonLayout}
              style={{ width: '100%', height: 280 }}
            />
          ) : (
            loadingPlaceholder
          )}
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium text-[var(--text-muted)]">
            UMAP {!umapResult && '(computing...)'}
          </p>
          {umapResult ? (
            <CanvasChart
              data={[makeTrace(umapResult)]}
              layout={commonLayout}
              style={{ width: '100%', height: 280 }}
            />
          ) : (
            loadingPlaceholder
          )}
        </div>
      </div>

        <div className="mt-3 rounded bg-[var(--surface-2,#27272a)] p-3 text-sm text-[var(--text-muted)]">
          The Swiss Roll is a 2D manifold embedded in 3D space. PCA projects linearly and
          fails to unroll the manifold — notice how colors (position along the roll) are
          mixed. t-SNE and UMAP preserve local neighborhoods and successfully unroll the
          structure.
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
