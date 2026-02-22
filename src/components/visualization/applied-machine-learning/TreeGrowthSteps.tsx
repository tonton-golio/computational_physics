"use client";

import React, { useMemo, useState } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationResults, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { mulberry32, gaussianPair } from './ml-utils';

interface Split {
  feature: 0 | 1;
  threshold: number;
  gini: number;
}

interface Region {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  indices: number[];
  split?: Split;
  label: number;
  gini: number;
}

function giniImpurity(labels: number[]): number {
  if (labels.length === 0) return 0;
  const p = labels.reduce((a, b) => a + b, 0) / labels.length;
  return 1 - p * p - (1 - p) * (1 - p);
}

function bestSplit(
  xs: number[],
  ys: number[],
  labels: number[],
  indices: number[],
): Split | null {
  if (indices.length < 4) return null;
  const curGini = giniImpurity(indices.map((i) => labels[i]));
  if (curGini < 0.01) return null;

  let bestScore = curGini;
  let best: Split | null = null;

  for (const feature of [0, 1] as const) {
    const vals = indices.map((i) => (feature === 0 ? xs[i] : ys[i]));
    const sorted = [...new Set(vals)].sort((a, b) => a - b);
    for (let t = 0; t < sorted.length - 1; t++) {
      const thresh = (sorted[t] + sorted[t + 1]) / 2;
      const left = indices.filter(
        (i) => (feature === 0 ? xs[i] : ys[i]) <= thresh,
      );
      const right = indices.filter(
        (i) => (feature === 0 ? xs[i] : ys[i]) > thresh,
      );
      if (left.length < 2 || right.length < 2) continue;
      const wg =
        (left.length / indices.length) *
          giniImpurity(left.map((i) => labels[i])) +
        (right.length / indices.length) *
          giniImpurity(right.map((i) => labels[i]));
      if (wg < bestScore - 0.001) {
        bestScore = wg;
        best = { feature, threshold: thresh, gini: wg };
      }
    }
  }
  return best;
}

function majorityLabel(labels: number[], indices: number[]): number {
  const sum = indices.reduce((a, i) => a + labels[i], 0);
  return sum / indices.length >= 0.5 ? 1 : 0;
}

export default function TreeGrowthSteps({}: SimulationComponentProps): React.ReactElement {
  const [step, setStep] = useState(0);
  const [seed, setSeed] = useState(42);

  // Generate data
  const { xs, ys, labels } = useMemo(() => {
    const rng = mulberry32(seed);
    const xArr: number[] = [];
    const yArr: number[] = [];
    const lArr: number[] = [];
    for (let i = 0; i < 120; i++) {
      const cls = i < 60 ? 0 : 1;
      const cx = cls === 0 ? 0.3 : 0.7;
      const cy = cls === 0 ? 0.35 : 0.65;
      const [g1, g2] = gaussianPair(rng);
      xArr.push(cx + g1 * 0.15);
      yArr.push(cy + g2 * 0.15);
      lArr.push(cls);
    }
    return { xs: xArr, ys: yArr, labels: lArr };
  }, [seed]);

  // Build tree step by step
  const regions = useMemo(() => {
    const allIndices = Array.from({ length: xs.length }, (_, i) => i);
    const initial: Region = {
      xMin: -0.1,
      xMax: 1.1,
      yMin: -0.1,
      yMax: 1.1,
      indices: allIndices,
      label: majorityLabel(labels, allIndices),
      gini: giniImpurity(allIndices.map((i) => labels[i])),
    };

    const history: Region[][] = [[initial]];
    let current = [initial];

    for (let s = 0; s < 5; s++) {
      // Find the region with highest impurity to split
      let bestRegionIdx = -1;
      let bestRegionGini = -1;
      for (let r = 0; r < current.length; r++) {
        if (current[r].gini > bestRegionGini && current[r].indices.length >= 4) {
          bestRegionGini = current[r].gini;
          bestRegionIdx = r;
        }
      }
      if (bestRegionIdx < 0 || bestRegionGini < 0.01) break;

      const region = current[bestRegionIdx];
      const split = bestSplit(xs, ys, labels, region.indices);
      if (!split) break;

      const leftIndices = region.indices.filter(
        (i) => (split.feature === 0 ? xs[i] : ys[i]) <= split.threshold,
      );
      const rightIndices = region.indices.filter(
        (i) => (split.feature === 0 ? xs[i] : ys[i]) > split.threshold,
      );

      const leftRegion: Region = {
        xMin: region.xMin,
        xMax: split.feature === 0 ? split.threshold : region.xMax,
        yMin: region.yMin,
        yMax: split.feature === 1 ? split.threshold : region.yMax,
        indices: leftIndices,
        label: majorityLabel(labels, leftIndices),
        gini: giniImpurity(leftIndices.map((i) => labels[i])),
      };
      const rightRegion: Region = {
        xMin: split.feature === 0 ? split.threshold : region.xMin,
        xMax: region.xMax,
        yMin: split.feature === 1 ? split.threshold : region.yMin,
        yMax: region.yMax,
        indices: rightIndices,
        label: majorityLabel(labels, rightIndices),
        gini: giniImpurity(rightIndices.map((i) => labels[i])),
      };

      const next = [...current];
      next.splice(bestRegionIdx, 1, leftRegion, rightRegion);
      // Store the split info on the parent for display
      region.split = split;
      current = next;
      history.push([...current]);
    }
    return history;
  }, [xs, ys, labels]);

  const maxStep = regions.length - 1;
  const currentRegions = regions[Math.min(step, maxStep)];

  // Build split-line traces
  const splitTraces = useMemo(() => {
    const traces: Array<{
      x: number[];
      y: number[];
      type: 'scatter';
      mode: 'lines';
      line: { color: string; width: number };
      showlegend: boolean;
    }> = [];
    // Collect all splits up to current step
    for (let s = 0; s < Math.min(step, maxStep); s++) {
      for (const r of regions[s]) {
        if (r.split) {
          if (r.split.feature === 0) {
            traces.push({
              x: [r.split.threshold, r.split.threshold],
              y: [r.yMin, r.yMax],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#fbbf24', width: 2.5 },
              showlegend: false,
            });
          } else {
            traces.push({
              x: [r.xMin, r.xMax],
              y: [r.split.threshold, r.split.threshold],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#fbbf24', width: 2.5 },
              showlegend: false,
            });
          }
        }
      }
    }
    return traces;
  }, [step, maxStep, regions]);

  const colors = labels.map((l) => (l === 0 ? '#3b82f6' : '#ef4444'));
  const avgGini =
    currentRegions.reduce((a, r) => a + r.gini * r.indices.length, 0) /
    xs.length;

  return (
    <SimulationPanel title="Decision Tree: Step-by-Step Growth">
      <SimulationSettings>
        <div className="flex flex-wrap items-center gap-3">
          <SimulationButton variant="secondary" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
            Previous
          </SimulationButton>
          <SimulationButton variant="primary" onClick={() => setStep((s) => Math.min(maxStep, s + 1))} disabled={step >= maxStep}>
            Next Split
          </SimulationButton>
          <SimulationButton variant="secondary" onClick={() => {
              setStep(0);
              setSeed((s) => s + 1);
            }}>
            New Data
          </SimulationButton>
        </div>
      </SimulationSettings>

      <SimulationMain>
        <CanvasChart
        data={[
          {
            x: xs,
            y: ys,
            type: 'scatter',
            mode: 'markers',
            marker: { color: colors, size: 6, opacity: 0.6 },
            name: 'Data',
            showlegend: false,
          },
          ...splitTraces,
        ]}
        layout={{
          xaxis: { title: { text: 'x\u2081' }, range: [-0.1, 1.1] },
          yaxis: { title: { text: 'x\u2082' }, range: [-0.1, 1.1] },
          margin: { t: 20, r: 20, b: 45, l: 55 },
          showlegend: false,
        }}
        style={{ width: '100%', height: 420 }}
      />

        <div className="mt-3 text-xs text-[var(--text-muted)]">
          Each step splits the region with the highest Gini impurity. Yellow lines show decision boundaries.
        </div>
      </SimulationMain>
      <SimulationResults>
        <span className="text-sm text-[var(--text-muted)]">
          Step {step}/{maxStep} | Regions: {currentRegions.length} | Avg Gini: {avgGini.toFixed(3)}
        </span>
      </SimulationResults>
    </SimulationPanel>
  );
}
