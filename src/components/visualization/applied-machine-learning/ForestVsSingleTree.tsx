'use client';

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { mulberry32, gaussianPair, linspace } from './ml-utils';

/**
 * Simple 1D decision stump: find the best split on xs to predict ys.
 * Returns a piecewise-constant prediction.
 */
function fitDeepTree(
  xs: number[],
  ys: number[],
  maxDepth: number,
): (x: number) => number {
  interface Node {
    threshold: number;
    left: Node | number;
    right: Node | number;
  }

  function buildNode(indices: number[], depth: number): Node | number {
    if (depth >= maxDepth || indices.length < 4) {
      return indices.reduce((a, i) => a + ys[i], 0) / (indices.length || 1);
    }
    let bestThresh = xs[indices[0]];
    let bestVar = Infinity;
    // Try a sample of thresholds
    const sorted = indices
      .map((i) => xs[i])
      .sort((a, b) => a - b);
    const step = Math.max(1, Math.floor(sorted.length / 20));
    for (let t = step; t < sorted.length; t += step) {
      const thresh = (sorted[t - 1] + sorted[t]) / 2;
      const leftIdx = indices.filter((i) => xs[i] <= thresh);
      const rightIdx = indices.filter((i) => xs[i] > thresh);
      if (leftIdx.length < 2 || rightIdx.length < 2) continue;
      const lMean =
        leftIdx.reduce((a, i) => a + ys[i], 0) / leftIdx.length;
      const rMean =
        rightIdx.reduce((a, i) => a + ys[i], 0) / rightIdx.length;
      const lVar = leftIdx.reduce((a, i) => a + (ys[i] - lMean) ** 2, 0);
      const rVar = rightIdx.reduce((a, i) => a + (ys[i] - rMean) ** 2, 0);
      const totalVar = lVar + rVar;
      if (totalVar < bestVar) {
        bestVar = totalVar;
        bestThresh = thresh;
      }
    }
    const leftIdx = indices.filter((i) => xs[i] <= bestThresh);
    const rightIdx = indices.filter((i) => xs[i] > bestThresh);
    if (leftIdx.length < 2 || rightIdx.length < 2) {
      return indices.reduce((a, i) => a + ys[i], 0) / indices.length;
    }
    return {
      threshold: bestThresh,
      left: buildNode(leftIdx, depth + 1),
      right: buildNode(rightIdx, depth + 1),
    };
  }

  function predict(node: Node | number, x: number): number {
    if (typeof node === 'number') return node;
    return x <= node.threshold
      ? predict(node.left, x)
      : predict(node.right, x);
  }

  const allIdx = Array.from({ length: xs.length }, (_, i) => i);
  const root = buildNode(allIdx, 0);
  return (x: number) => predict(root, x);
}

export default function ForestVsSingleTree(): React.ReactElement {
  const [nTrees, setNTrees] = useState(20);
  const [noiseLevel, setNoiseLevel] = useState(1.5);
  const [seed, setSeed] = useState(42);

  // Generate noisy sine data
  const { xs, ys } = useMemo(() => {
    const rng = mulberry32(seed);
    const n = 80;
    const xArr: number[] = [];
    const yArr: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = rng() * 10;
      const [noise] = gaussianPair(rng);
      xArr.push(x);
      yArr.push(Math.sin(x) + noise * noiseLevel * 0.3);
    }
    return { xs: xArr, ys: yArr };
  }, [noiseLevel, seed]);

  // Single deep tree
  const singleTreePred = useMemo(() => {
    const predict = fitDeepTree(xs, ys, 10);
    const xEval = linspace(0, 10, 200);
    return { x: xEval, y: xEval.map(predict) };
  }, [xs, ys]);

  // Random forest: bootstrap + shallow trees
  const forestPred = useMemo(() => {
    const xEval = linspace(0, 10, 200);
    const predictions = Array.from({ length: xEval.length }, () => 0);

    for (let t = 0; t < nTrees; t++) {
      const rng = mulberry32(seed + t * 137 + 7);
      // Bootstrap sample
      const bootIdx: number[] = [];
      for (let i = 0; i < xs.length; i++) {
        bootIdx.push(Math.floor(rng() * xs.length));
      }
      const bootX = bootIdx.map((i) => xs[i]);
      const bootY = bootIdx.map((i) => ys[i]);
      const predict = fitDeepTree(bootX, bootY, 4);
      for (let i = 0; i < xEval.length; i++) {
        predictions[i] += predict(xEval[i]) / nTrees;
      }
    }
    return { x: xEval, y: predictions };
  }, [xs, ys, nTrees, seed]);

  // True function
  const trueLine = useMemo(() => {
    const xEval = linspace(0, 10, 200);
    return { x: xEval, y: xEval.map((x) => Math.sin(x)) };
  }, []);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Single Deep Tree vs Random Forest
      </h3>

      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <label className="text-sm text-[var(--text-muted)]">
            Trees in forest: {nTrees}
          </label>
          <Slider
            min={1}
            max={100}
            step={1}
            value={[nTrees]}
            onValueChange={([v]) => setNTrees(v)}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">
            Noise level: {noiseLevel.toFixed(1)}
          </label>
          <Slider
            min={0.2}
            max={3}
            step={0.1}
            value={[noiseLevel]}
            onValueChange={([v]) => setNoiseLevel(v)}
          />
        </div>
        <div className="flex items-end">
          <button
            onClick={() => setSeed((s) => s + 1)}
            className="rounded bg-[var(--surface-2,#27272a)] px-4 py-1.5 text-sm font-medium text-[var(--text-strong)] hover:opacity-90"
          >
            New Data
          </button>
        </div>
      </div>

      <CanvasChart
        data={[
          {
            x: xs,
            y: ys,
            type: 'scatter',
            mode: 'markers',
            marker: { color: '#64748b', size: 5, opacity: 0.5 },
            name: 'Noisy data',
          },
          {
            x: trueLine.x,
            y: trueLine.y,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#6366f1', width: 2, dash: 'dash' },
            name: 'True f(x)=sin(x)',
          },
          {
            x: singleTreePred.x,
            y: singleTreePred.y,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ef4444', width: 2 },
            name: 'Single tree (depth 10)',
          },
          {
            x: forestPred.x,
            y: forestPred.y,
            type: 'scatter',
            mode: 'lines',
            line: { color: '#10b981', width: 2.5 },
            name: `Forest (${nTrees} trees)`,
          },
        ]}
        layout={{
          xaxis: { title: { text: 'x' } },
          yaxis: { title: { text: 'y' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 420 }}
      />

      <div className="mt-3 text-xs text-[var(--text-muted)]">
        The red single tree overfits with jagged predictions. The green forest averages many shallow trees, producing a smoother curve closer to the true function (dashed).
      </div>
    </div>
  );
}
