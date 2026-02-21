'use client';

import React, { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';

function computeTransferMatrix(J: number, tRange: number[]) {
  const xi: number[] = [];
  const ratio: number[] = [];

  for (const T of tRange) {
    const beta = 1 / T;
    const K = beta * J;
    const lambdaPlus = 2 * Math.cosh(K);
    const lambdaMinus = 2 * Math.sinh(K);
    const r = Math.abs(lambdaMinus / lambdaPlus);
    ratio.push(r);
    if (r > 0 && r < 1) {
      xi.push(-1 / Math.log(r));
    } else {
      xi.push(0);
    }
  }

  return { xi, ratio };
}

export function TransferMatrixDemo() {
  const [J, setJ] = useState(1.0);
  const [currentT, setCurrentT] = useState(2.0);

  const tRange = useMemo(() => {
    const pts: number[] = [];
    for (let i = 1; i <= 200; i++) {
      pts.push(0.05 * i);
    }
    return pts;
  }, []);

  const data = useMemo(() => computeTransferMatrix(J, tRange), [J, tRange]);

  const currentXi = useMemo(() => {
    const beta = 1 / currentT;
    const K = beta * J;
    const r = Math.abs(2 * Math.sinh(K) / (2 * Math.cosh(K)));
    return r > 0 && r < 1 ? -1 / Math.log(r) : 0;
  }, [J, currentT]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">
            Coupling J: {J.toFixed(2)}
          </label>
          <Slider
            min={0.1}
            max={3}
            step={0.1}
            value={[J]}
            onValueChange={([v]) => setJ(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">
            Temperature T: {currentT.toFixed(2)} ({'\u03BE'} = {currentXi.toFixed(2)})
          </label>
          <Slider
            min={0.1}
            max={10}
            step={0.1}
            value={[currentT]}
            onValueChange={([v]) => setCurrentT(v)}
            className="w-full"
          />
        </div>
      </div>

      <CanvasChart
        data={[
          {
            x: tRange,
            y: data.xi,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#8b5cf6', width: 2 },
            name: '\u03BE(T)',
          },
          {
            x: [currentT, currentT],
            y: [0, Math.max(...data.xi)],
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#ef4444', width: 1.5 },
            name: `T = ${currentT.toFixed(1)}`,
          },
        ]}
        layout={{
          title: { text: 'Correlation Length \u03BE(T) from Transfer Matrix', font: { size: 14 } },
          xaxis: { title: { text: 'Temperature T' } },
          yaxis: { title: { text: '\u03BE' } },
          showlegend: true,
          margin: { t: 40, r: 20, b: 50, l: 60 },
        }}
        style={{ width: '100%', height: 400 }}
      />

      <p className="text-sm text-[var(--text-muted)]">
        The correlation length {'\u03BE'} = -1/ln({'\u03BB'}{'\u208B'}/{'\u03BB'}{'\u208A'}) grows as T {'\u2192'} 0 but never diverges at finite T — confirming no phase transition in 1D.
      </p>
    </div>
  );
}
