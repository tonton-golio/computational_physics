"use client";

import React, { useMemo, useState } from 'react';
import { clamp } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { mulberry32, gaussianPair, CLUSTER_COLORS, linspace } from './ml-utils';

// ── Minimal CART decision tree ──────────────────────────────────────────

interface TreeNode {
  feature?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  value?: number; // leaf prediction (probability of class 1)
}

function giniImpurity(labels: number[]): number {
  if (labels.length === 0) return 0;
  const p = labels.reduce((a, b) => a + b, 0) / labels.length;
  return 1 - p * p - (1 - p) * (1 - p);
}

function buildTree(
  X: number[][],
  y: number[],
  maxDepth: number,
  depth: number = 0,
  rng?: () => number,
  featureSubset?: boolean,
): TreeNode {
  if (depth >= maxDepth || y.length <= 2) {
    return { value: y.length > 0 ? y.reduce((a, b) => a + b, 0) / y.length : 0.5 };
  }

  const uniqueLabels = new Set(y);
  if (uniqueLabels.size === 1) {
    return { value: y[0] };
  }

  let bestGain = -Infinity;
  let bestFeature = 0;
  let bestThreshold = 0;

  const parentImpurity = giniImpurity(y);
  const features = featureSubset && rng
    ? [Math.floor(rng() * 2)] // random feature subset for RF
    : [0, 1];

  for (const f of features) {
    const values = X.map((row) => row[f]);
    const sorted = [...new Set(values)].sort((a, b) => a - b);
    const step = Math.max(1, Math.floor(sorted.length / 20));

    for (let i = 0; i < sorted.length - 1; i += step) {
      const threshold = (sorted[i] + sorted[i + 1]) / 2;
      const leftIdx: number[] = [];
      const rightIdx: number[] = [];

      for (let j = 0; j < X.length; j++) {
        if (X[j][f] <= threshold) leftIdx.push(j);
        else rightIdx.push(j);
      }

      if (leftIdx.length === 0 || rightIdx.length === 0) continue;

      const leftLabels = leftIdx.map((j) => y[j]);
      const rightLabels = rightIdx.map((j) => y[j]);
      const gain =
        parentImpurity -
        (leftLabels.length / y.length) * giniImpurity(leftLabels) -
        (rightLabels.length / y.length) * giniImpurity(rightLabels);

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = f;
        bestThreshold = threshold;
      }
    }
  }

  if (bestGain <= 0) {
    return { value: y.reduce((a, b) => a + b, 0) / y.length };
  }

  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (let j = 0; j < X.length; j++) {
    if (X[j][bestFeature] <= bestThreshold) leftIdx.push(j);
    else rightIdx.push(j);
  }

  return {
    feature: bestFeature,
    threshold: bestThreshold,
    left: buildTree(
      leftIdx.map((j) => X[j]),
      leftIdx.map((j) => y[j]),
      maxDepth,
      depth + 1,
      rng,
      featureSubset,
    ),
    right: buildTree(
      rightIdx.map((j) => X[j]),
      rightIdx.map((j) => y[j]),
      maxDepth,
      depth + 1,
      rng,
      featureSubset,
    ),
  };
}

function predictTree(tree: TreeNode, x: number[]): number {
  if (tree.value !== undefined && tree.feature === undefined) return tree.value;
  if (x[tree.feature!] <= tree.threshold!) {
    return predictTree(tree.left!, x);
  }
  return predictTree(tree.right!, x);
}

// ── Data generation ─────────────────────────────────────────────────────

function generateXorData(
  n: number,
  noise: number,
  seed: number,
): { X: number[][]; y: number[] } {
  const rng = mulberry32(seed);
  const X: number[][] = [];
  const y: number[] = [];

  for (let i = 0; i < n; i++) {
    const a = rng() > 0.5 ? 1 : 0;
    const b = rng() > 0.5 ? 1 : 0;
    const label = a !== b ? 1 : 0;
    const [nx, ny] = gaussianPair(rng);
    X.push([a + nx * noise, b + ny * noise]);
    y.push(label);
  }
  return { X, y };
}

// ── Component ───────────────────────────────────────────────────────────

type Method = 'single' | 'rf' | 'gbt';

export default function XorEnsembleArena({}: SimulationComponentProps): React.ReactElement {
  const [method, setMethod] = useState<Method>('rf');
  const [maxDepth, setMaxDepth] = useState(5);
  const [nTrees, setNTrees] = useState(50);
  const [addNoise, setAddNoise] = useState(false);
  const noise = addNoise ? 0.35 : 0.2;

  const { X, y } = useMemo(() => generateXorData(400, noise, 42), [noise]);

  // Train model(s)
  const predict = useMemo(() => {
    if (method === 'single') {
      const tree = buildTree(X, y, maxDepth);
      return (x: number[]) => predictTree(tree, x);
    }

    if (method === 'rf') {
      // Random Forest: bagging + random feature subset
      const trees: TreeNode[] = [];
      for (let t = 0; t < nTrees; t++) {
        const rng = mulberry32(t * 137 + 7);
        // Bootstrap sample
        const idx = Array.from({ length: X.length }, () => Math.floor(rng() * X.length));
        const Xb = idx.map((i) => X[i]);
        const yb = idx.map((i) => y[i]);
        trees.push(buildTree(Xb, yb, maxDepth, 0, mulberry32(t * 31 + 11), true));
      }
      return (x: number[]) => {
        const avg = trees.reduce((a, tree) => a + predictTree(tree, x), 0) / trees.length;
        return avg;
      };
    }

    // Gradient Boosted Trees (simplified)
    const lr = 0.3;
    let residuals = [...y];
    const stumps: { tree: TreeNode; weight: number }[] = [];
    let predictions = new Array(X.length).fill(0.5);

    for (let t = 0; t < nTrees; t++) {
      // Residuals = y - predictions (pseudo-residuals for squared loss)
      residuals = y.map((yi, i) => yi - predictions[i]);
      const tree = buildTree(X, residuals, Math.min(maxDepth, 3), 0, mulberry32(t * 71 + 3));
      stumps.push({ tree, weight: lr });
      predictions = predictions.map((p, i) => {
        const pred = predictTree(tree, X[i]);
        return clamp(p + lr * pred, 0, 1);
      });
    }

    return (x: number[]) => {
      let pred = 0.5;
      for (const { tree, weight } of stumps) {
        pred += weight * predictTree(tree, x);
      }
      return clamp(pred, 0, 1);
    };
  }, [X, y, method, maxDepth, nTrees]);

  // Decision boundary heatmap
  const boundary = useMemo(() => {
    const res = 50;
    const xs = linspace(-0.5, 1.5, res);
    const ys = linspace(-0.5, 1.5, res);
    const z = ys.map((yy) => xs.map((xx) => predict([xx, yy])));
    return { x: xs, y: ys, z };
  }, [predict]);

  // Accuracy
  const accuracy = useMemo(() => {
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      const pred = predict(X[i]) > 0.5 ? 1 : 0;
      if (pred === y[i]) correct++;
    }
    return correct / X.length;
  }, [X, y, predict]);

  const groundColors = y.map((l) => (l === 0 ? CLUSTER_COLORS[0] : CLUSTER_COLORS[3]));

  return (
    <SimulationPanel title="XOR Ensemble Arena">
      <SimulationSettings>
        <div className="flex flex-wrap gap-2">
          {(['single', 'rf', 'gbt'] as Method[]).map((m) => (
            <SimulationButton
              key={m}
              variant={method === m ? 'primary' : 'secondary'}
              onClick={() => setMethod(m)}
            >
              {m === 'single' ? 'Single Tree' : m === 'rf' ? 'Random Forest' : 'Gradient Boosted'}
            </SimulationButton>
          ))}
        </div>
        <SimulationLabel className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={addNoise}
            onChange={(e) => setAddNoise(e.target.checked)}
          />
          Add noise
        </SimulationLabel>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-2">
          <div>
            <SimulationLabel>Max depth: {maxDepth}</SimulationLabel>
            <Slider min={1} max={15} step={1} value={[maxDepth]} onValueChange={([v]) => setMaxDepth(v)} />
          </div>
          {method !== 'single' && (
            <div>
              <SimulationLabel>Trees: {nTrees}</SimulationLabel>
              <Slider min={5} max={100} step={5} value={[nTrees]} onValueChange={([v]) => setNTrees(v)} />
            </div>
          )}
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Decision boundary</p>
          <CanvasHeatmap
            data={[{ z: boundary.z, colorscale: 'RdBu', showscale: false, zmin: 0, zmax: 1 }]}
            layout={{
              xaxis: { title: { text: 'x\u2081' } },
              yaxis: { title: { text: 'x\u2082' } },
              margin: { t: 20, r: 20, b: 45, l: 55 },
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Ground truth</p>
          <CanvasChart
            data={[
              {
                x: X.map((r) => r[0]),
                y: X.map((r) => r[1]),
                type: 'scatter',
                mode: 'markers',
                marker: { color: groundColors, size: 5, opacity: 0.6 },
                name: 'Data',
              },
            ]}
            layout={{
              xaxis: { title: { text: 'x\u2081' }, range: [-0.5, 1.5] },
              yaxis: { title: { text: 'x\u2082' }, range: [-0.5, 1.5] },
              margin: { t: 20, r: 20, b: 45, l: 55 },
              showlegend: false,
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
      </div>

        {addNoise && method === 'single' && maxDepth > 8 && (
          <div className="mt-3 rounded bg-red-900/30 p-3 text-sm text-red-300">
            The single tree is overfitting to noise. Notice the jagged decision boundary.
            Switch to Random Forest to see how ensembles stay stable.
          </div>
        )}
        {method === 'rf' && accuracy > 0.9 && (
          <div className="mt-3 rounded bg-emerald-900/30 p-3 text-sm text-emerald-300">
            The Random Forest handles the XOR pattern by combining many weak learners.
            Each tree sees a different bootstrap sample and feature subset.
          </div>
        )}
      </SimulationMain>
      <SimulationResults>
        <div className="rounded bg-[var(--surface-2,#27272a)] px-3 py-1.5 text-sm font-medium text-[var(--text-strong)]">
          Accuracy: {(accuracy * 100).toFixed(1)}%
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
