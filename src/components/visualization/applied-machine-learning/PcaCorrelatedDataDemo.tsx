"use client";

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { gaussianNoise } from './ml-utils';

export default function PcaCorrelatedDataDemo({}: SimulationComponentProps): React.ReactElement {
  const [corr, setCorr] = useState(0.8);
  const [n, setN] = useState(250);
  const data = useMemo(() => {
    const x = Array.from({ length: n }, (_, i) => gaussianNoise(i * 2.7 + 19, 1));
    const y = x.map((xi, i) => corr * xi + gaussianNoise(i * 2.9 + 23, Math.sqrt(Math.max(1 - corr * corr, 0.05))));
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;
    const xc = x.map((v) => v - meanX);
    const yc = y.map((v) => v - meanY);
    const sxx = xc.reduce((a, b) => a + b * b, 0) / (n - 1);
    const syy = yc.reduce((a, b) => a + b * b, 0) / (n - 1);
    const sxy = xc.reduce((a, b, i) => a + b * yc[i], 0) / (n - 1);
    const tr = sxx + syy;
    const det = sxx * syy - sxy * sxy;
    const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
    const lambda1 = tr / 2 + disc;
    const lambda2 = tr / 2 - disc;
    const v1 = Math.abs(sxy) > 1e-8 ? [lambda1 - syy, sxy] : [1, 0];
    const norm = Math.hypot(v1[0], v1[1]) || 1;
    return { x, y, pcx: v1[0] / norm, pcy: v1[1] / norm, lambda1, lambda2 };
  }, [corr, n]);

  const evr1 = (data.lambda1 / (data.lambda1 + data.lambda2)).toFixed(2);

  return (
    <SimulationPanel title="PCA on Correlated Data">
      <SimulationConfig>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <SimulationLabel>Correlation: {corr.toFixed(2)}</SimulationLabel>
            <Slider min={0.1} max={0.98} step={0.01} value={[corr]} onValueChange={([v]) => setCorr(v)} />
          </div>
          <div>
            <SimulationLabel>Samples: {n}</SimulationLabel>
            <Slider min={100} max={600} step={20} value={[n]} onValueChange={([v]) => setN(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
          data={[
            { x: data.x, y: data.y, type: 'scatter', mode: 'markers', marker: { color: '#3b82f6', size: 6, opacity: 0.6 }, name: 'Data' },
            { x: [0, 2.5 * data.pcx], y: [0, 2.5 * data.pcy], type: 'scatter', mode: 'lines', line: { color: '#ec4899', width: 4 }, name: 'PC1' },
            { x: [0, -2.5 * data.pcy], y: [0, 2.5 * data.pcx], type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 4 }, name: 'PC2' },
          ]}
          layout={{
            xaxis: { title: { text: 'x1' } },
            yaxis: { title: { text: 'x2' } },
            margin: { t: 20, r: 20, b: 45, l: 55 },
          }}
          style={{ width: '100%', height: 360 }}
        />
      </SimulationMain>
      <SimulationResults>
        <div className="text-right text-sm text-[var(--text-muted)]">EVR1: {evr1}</div>
      </SimulationResults>
    </SimulationPanel>
  );
}
