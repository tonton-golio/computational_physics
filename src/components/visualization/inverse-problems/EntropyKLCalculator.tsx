'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const N_BINS = 6;
const LABELS = ['A', 'B', 'C', 'D', 'E', 'F'];

function normalize(arr: number[]): number[] {
  const s = arr.reduce((a, b) => a + b, 0);
  return s > 0 ? arr.map(v => v / s) : arr.map(() => 1 / arr.length);
}

function shannonEntropy(p: number[]): number {
  let h = 0;
  for (const pi of p) if (pi > 1e-12) h -= pi * Math.log2(pi);
  return h;
}

function klDivergence(p: number[], q: number[]): number {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 1e-12 && q[i] > 1e-12) {
      kl += p[i] * Math.log2(p[i] / q[i]);
    } else if (p[i] > 1e-12) {
      return Infinity;
    }
  }
  return kl;
}

export default function EntropyKLCalculator({ id: _id }: SimulationComponentProps) {
  // Control distributions via "concentration" sliders: each adjusts relative weight
  const [pWeights, setPWeights] = useState([3, 2, 4, 1, 2, 3]);
  const [qWeights, setQWeights] = useState([1, 1, 1, 1, 1, 1]);

  const handleP = (idx: number, val: number) => {
    const next = [...pWeights];
    next[idx] = val;
    setPWeights(next);
  };
  const handleQ = (idx: number, val: number) => {
    const next = [...qWeights];
    next[idx] = val;
    setQWeights(next);
  };

  const result = useMemo(() => {
    const p = normalize(pWeights);
    const q = normalize(qWeights);

    const hP = shannonEntropy(p);
    const hQ = shannonEntropy(q);
    const klPQ = klDivergence(p, q);
    const klQP = klDivergence(q, p);

    // Cross-entropy H(P, Q) = H(P) + D_KL(P||Q)
    const crossPQ = hP + klPQ;
    const crossQP = hQ + klQP;

    // Jensen-Shannon divergence (symmetric)
    const m = p.map((pi, i) => 0.5 * (pi + q[i]));
    const jsd = 0.5 * klDivergence(p, m) + 0.5 * klDivergence(q, m);

    const maxEntropy = Math.log2(N_BINS);

    return { p, q, hP, hQ, klPQ, klQP, crossPQ, crossQP, jsd, maxEntropy };
  }, [pWeights, qWeights]);

  const xIdx = Array.from({ length: N_BINS }, (_, i) => i);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Entropy and Divergence Calculator
      </h3>

      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <p className="text-sm font-medium text-blue-400 mb-2">Distribution P</p>
          <div className="grid grid-cols-3 gap-2">
            {LABELS.map((label, i) => (
              <div key={`p-${i}`}>
                <label className="text-xs text-[var(--text-muted)]">{label}: {pWeights[i]}</label>
                <Slider value={[pWeights[i]]} onValueChange={([v]) => handleP(i, v)}
                  min={0} max={10} step={1} />
              </div>
            ))}
          </div>
        </div>
        <div>
          <p className="text-sm font-medium text-red-400 mb-2">Distribution Q</p>
          <div className="grid grid-cols-3 gap-2">
            {LABELS.map((label, i) => (
              <div key={`q-${i}`}>
                <label className="text-xs text-[var(--text-muted)]">{label}: {qWeights[i]}</label>
                <Slider value={[qWeights[i]]} onValueChange={([v]) => handleQ(i, v)}
                  min={0} max={10} step={1} />
              </div>
            ))}
          </div>
        </div>
      </div>

      <CanvasChart
        data={[
          {
            x: xIdx, y: result.p,
            type: 'bar', marker: { color: 'rgba(59,130,246,0.7)' },
            name: 'P',
          },
          {
            x: xIdx, y: result.q,
            type: 'bar', marker: { color: 'rgba(239,68,68,0.7)' },
            name: 'Q',
          },
        ]}
        layout={{
          title: { text: 'Distributions P and Q' },
          barmode: 'group',
          xaxis: { title: { text: 'Outcome' }, tickvals: xIdx, ticktext: LABELS },
          yaxis: { title: { text: 'Probability' }, range: [0, 1] },
          height: 300,
          margin: { t: 40, b: 50, l: 50, r: 20 },
        }}
        style={{ width: '100%' }}
      />

      <div className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="rounded p-3 border border-[var(--border-strong)]">
          <div className="text-xs text-[var(--text-muted)]">H(P)</div>
          <div className="text-lg font-mono text-blue-400">{result.hP.toFixed(4)}</div>
          <div className="text-xs text-[var(--text-muted)]">
            / {result.maxEntropy.toFixed(2)} max
          </div>
        </div>
        <div className="rounded p-3 border border-[var(--border-strong)]">
          <div className="text-xs text-[var(--text-muted)]">H(Q)</div>
          <div className="text-lg font-mono text-red-400">{result.hQ.toFixed(4)}</div>
          <div className="text-xs text-[var(--text-muted)]">
            / {result.maxEntropy.toFixed(2)} max
          </div>
        </div>
        <div className="rounded p-3 border border-[var(--border-strong)]">
          <div className="text-xs text-[var(--text-muted)]">D_KL(P || Q)</div>
          <div className="text-lg font-mono text-[var(--text-strong)]">
            {isFinite(result.klPQ) ? result.klPQ.toFixed(4) : '\u221E'}
          </div>
        </div>
        <div className="rounded p-3 border border-[var(--border-strong)]">
          <div className="text-xs text-[var(--text-muted)]">D_KL(Q || P)</div>
          <div className="text-lg font-mono text-[var(--text-strong)]">
            {isFinite(result.klQP) ? result.klQP.toFixed(4) : '\u221E'}
          </div>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-3">
        <div className="rounded p-3 border border-[var(--border-strong)]">
          <div className="text-xs text-[var(--text-muted)]">Cross-entropy H(P,Q)</div>
          <div className="font-mono text-sm text-[var(--text-strong)]">
            {isFinite(result.crossPQ) ? result.crossPQ.toFixed(4) : '\u221E'}
          </div>
        </div>
        <div className="rounded p-3 border border-[var(--border-strong)]">
          <div className="text-xs text-[var(--text-muted)]">Cross-entropy H(Q,P)</div>
          <div className="font-mono text-sm text-[var(--text-strong)]">
            {isFinite(result.crossQP) ? result.crossQP.toFixed(4) : '\u221E'}
          </div>
        </div>
        <div className="rounded p-3 border border-[var(--border-strong)]">
          <div className="text-xs text-[var(--text-muted)]">Jensen-Shannon (symmetric)</div>
          <div className="font-mono text-sm text-[var(--text-strong)]">{result.jsd.toFixed(4)}</div>
        </div>
      </div>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          Make P and Q identical and verify all divergences go to zero. Set P to a spike (one
          bin high, rest zero) and watch H(P) drop to zero while KL divergence grows. Compare
          D_KL(P||Q) and D_KL(Q||P) to see the asymmetry. The Jensen-Shannon divergence is
          always symmetric and bounded. Setting any Q bin to zero where P is non-zero sends
          D_KL(P||Q) to infinity, reflecting the severity of assigning zero probability to
          possible events.
        </p>
      </div>
    </div>
  );
}
