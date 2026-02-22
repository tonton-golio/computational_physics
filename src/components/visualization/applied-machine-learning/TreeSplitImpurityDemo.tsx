"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { gaussianNoise } from './ml-utils';

export default function TreeSplitImpurityDemo({}: SimulationComponentProps): React.ReactElement {
  const [threshold, setThreshold] = useState(0.5);
  const [splitFeature, setSplitFeature] = useState<0 | 1>(0);

  // Generate a 2D dataset with two clusters
  const data = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < 150; i++) {
      const c = i < 75 ? 0 : 1;
      xs.push((c === 0 ? 0.3 : 0.7) + gaussianNoise(i * 2.1 + 5, 0.15));
      ys.push((c === 0 ? 0.35 : 0.65) + gaussianNoise(i * 3.3 + 11, 0.15));
      labels.push(c);
    }
    return { xs, ys, labels };
  }, []);

  // Compute impurity for the current split
  const { leftGini, rightGini, leftEntropy, rightEntropy, weightedGini, weightedEntropy, leftCount, rightCount } =
    useMemo(() => {
      const featureVals = splitFeature === 0 ? data.xs : data.ys;
      const leftLabels: number[] = [];
      const rightLabels: number[] = [];
      for (let i = 0; i < featureVals.length; i++) {
        if (featureVals[i] <= threshold) leftLabels.push(data.labels[i]);
        else rightLabels.push(data.labels[i]);
      }
      const gini = (ls: number[]) => {
        if (ls.length === 0) return 0;
        const p = ls.reduce((a, b) => a + b, 0) / ls.length;
        return 1 - p * p - (1 - p) * (1 - p);
      };
      const ent = (ls: number[]) => {
        if (ls.length === 0) return 0;
        const p = ls.reduce((a, b) => a + b, 0) / ls.length;
        if (p < 1e-9 || p > 1 - 1e-9) return 0;
        return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
      };
      const n = featureVals.length;
      const lg = gini(leftLabels);
      const rg = gini(rightLabels);
      const le = ent(leftLabels);
      const re = ent(rightLabels);
      return {
        leftGini: lg,
        rightGini: rg,
        leftEntropy: le,
        rightEntropy: re,
        weightedGini: (leftLabels.length / n) * lg + (rightLabels.length / n) * rg,
        weightedEntropy: (leftLabels.length / n) * le + (rightLabels.length / n) * re,
        leftCount: leftLabels.length,
        rightCount: rightLabels.length,
      };
    }, [data, threshold, splitFeature]);

  const colors = data.labels.map((l) => (l === 0 ? '#3b82f6' : '#ef4444'));
  const featureLabel = splitFeature === 0 ? 'x\u2081' : 'x\u2082';

  return (
    <SimulationPanel title="Tree Split & Impurity">
      <SimulationSettings>
        <div>
          <SimulationLabel>Split feature</SimulationLabel>
          <div className="mt-1 flex gap-2">
            <SimulationButton
              variant={splitFeature === 0 ? 'primary' : 'secondary'}
              onClick={() => setSplitFeature(0)}
            >
              x&#8321;
            </SimulationButton>
            <SimulationButton
              variant={splitFeature === 1 ? 'primary' : 'secondary'}
              onClick={() => setSplitFeature(1)}
            >
              x&#8322;
            </SimulationButton>
          </div>
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Split threshold ({featureLabel}): {threshold.toFixed(2)}
          </SimulationLabel>
          <Slider min={0} max={1} step={0.01} value={[threshold]} onValueChange={([v]) => setThreshold(v)} />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
        data={[
          {
            x: data.xs,
            y: data.ys,
            type: 'scatter',
            mode: 'markers',
            marker: { color: colors, size: 7, opacity: 0.7 },
            name: 'Data',
          },
          // Threshold line
          ...(splitFeature === 0
            ? [{
                x: [threshold, threshold],
                y: [0, 1],
                type: 'scatter' as const,
                mode: 'lines' as const,
                line: { color: '#fbbf24', width: 3 },
                name: 'Split',
              }]
            : [{
                x: [0, 1] as number[],
                y: [threshold, threshold],
                type: 'scatter' as const,
                mode: 'lines' as const,
                line: { color: '#fbbf24', width: 3 },
                name: 'Split',
              }]),
        ]}
        layout={{
          xaxis: { title: { text: 'x\u2081' }, range: [-0.1, 1.1] },
          yaxis: { title: { text: 'x\u2082' }, range: [-0.1, 1.1] },
          margin: { t: 20, r: 20, b: 45, l: 55 },
          showlegend: false,
        }}
        style={{ width: '100%', height: 360 }}
      />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          <div>Left ({leftCount}): Gini={leftGini.toFixed(3)}, H={leftEntropy.toFixed(3)}</div>
          <div>Right ({rightCount}): Gini={rightGini.toFixed(3)}, H={rightEntropy.toFixed(3)}</div>
          <div className="mt-1 font-medium text-[var(--text-strong)]">
            Weighted Gini: {weightedGini.toFixed(4)} | Entropy: {weightedEntropy.toFixed(4)}
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
