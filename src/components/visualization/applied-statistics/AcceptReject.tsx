'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

// Target: Beta(2,5) density; Proposal: Uniform(0,1) * M
function betaPDF(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return 0;
  // Use log-gamma approximation for B(a,b)
  const logB = lgamma(a) + lgamma(b) - lgamma(a + b);
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
}

function lgamma(z: number): number {
  // Stirling approximation for positive z
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1;
  const g = 7;
  const coef = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  let x = coef[0];
  for (let i = 1; i < g + 2; i++) x += coef[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

export default function AcceptReject() {
  const [numSamples, setNumSamples] = useState(200);

  const { targetX, targetY, acceptedX, acceptedY, rejectedX, rejectedY, efficiency } = useMemo(() => {
    const a = 2, b = 5;
    // Find M: max of betaPDF on [0,1]
    const M = betaPDF((a - 1) / (a + b - 2), a, b) * 1.05;

    const tx: number[] = [];
    const ty: number[] = [];
    for (let i = 0; i <= 200; i++) {
      const x = i / 200;
      tx.push(x);
      ty.push(betaPDF(x, a, b));
    }

    const accX: number[] = [];
    const accY: number[] = [];
    const rejX: number[] = [];
    const rejY: number[] = [];

    // Seed random for reproducibility within useMemo
    let totalTrials = 0;
    let accepted = 0;
    while (accepted < numSamples && totalTrials < numSamples * 20) {
      const x = Math.random();
      const u = Math.random() * M;
      totalTrials++;
      if (u <= betaPDF(x, a, b)) {
        accX.push(x);
        accY.push(u);
        accepted++;
      } else {
        rejX.push(x);
        rejY.push(u);
      }
    }

    return {
      targetX: tx, targetY: ty,
      acceptedX: accX, acceptedY: accY,
      rejectedX: rejX, rejectedY: rejY,
      efficiency: totalTrials > 0 ? (accepted / totalTrials * 100) : 0,
    };
  }, [numSamples]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Accept-Reject Sampling</h3>
      <div className="grid grid-cols-1 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Number of accepted samples: {numSamples}</label>
          <Slider value={[numSamples]} onValueChange={([v]) => setNumSamples(v)} min={20} max={1000} step={10} />
        </div>
      </div>
      <div className="mb-3 text-sm text-[var(--text-muted)]">
        Target: Beta(2, 5) | Proposal: Uniform | Acceptance rate: {efficiency.toFixed(1)}%
      </div>
      <CanvasChart
        data={[
          {
            x: rejectedX, y: rejectedY, type: 'scatter', mode: 'markers',
            marker: { color: '#ef4444', size: 3, opacity: 0.3 }, name: 'Rejected',
          },
          {
            x: acceptedX, y: acceptedY, type: 'scatter', mode: 'markers',
            marker: { color: '#10b981', size: 3, opacity: 0.6 }, name: 'Accepted',
          },
          {
            x: targetX, y: targetY, type: 'scatter', mode: 'lines',
            line: { color: '#3b82f6', width: 2.5 }, name: 'Target PDF',
          },
        ]}
        layout={{
          height: 400,
          xaxis: { title: { text: 'x' }, range: [0, 1] },
          yaxis: { title: { text: 'Density / envelope height' } },
        }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
