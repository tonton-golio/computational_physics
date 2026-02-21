'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

// Integrate sin(pi*x) from 0 to 1, exact answer = 2/pi ~ 0.6366
export default function MonteCarloIntegrationStats() {
  const [numSamples, setNumSamples] = useState(500);

  const { insideX, insideY, outsideX, outsideY, curveX, curveY, estimates, trueVal } = useMemo(() => {
    const f = (x: number) => Math.sin(Math.PI * x);
    const exact = 2 / Math.PI;
    const inX: number[] = [];
    const inY: number[] = [];
    const outX: number[] = [];
    const outY: number[] = [];

    let runningSum = 0;
    const est: number[] = [];
    const estN: number[] = [];

    for (let i = 0; i < numSamples; i++) {
      const x = Math.random();
      const y = Math.random();
      const fx = f(x);
      if (y <= fx) {
        inX.push(x);
        inY.push(y);
      } else {
        outX.push(x);
        outY.push(y);
      }
      runningSum += fx;
      if ((i + 1) % 5 === 0 || i === numSamples - 1) {
        est.push(runningSum / (i + 1));
        estN.push(i + 1);
      }
    }

    const cx: number[] = [];
    const cy: number[] = [];
    for (let i = 0; i <= 100; i++) {
      const x = i / 100;
      cx.push(x);
      cy.push(f(x));
    }

    return {
      insideX: inX, insideY: inY,
      outsideX: outX, outsideY: outY,
      curveX: cx, curveY: cy,
      estimates: { n: estN, val: est },
      trueVal: exact,
    };
  }, [numSamples]);

  const lastEstimate = estimates.val[estimates.val.length - 1] || 0;

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Monte Carlo Integration</h3>
      <div className="grid grid-cols-1 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Samples: {numSamples}</label>
          <Slider value={[numSamples]} onValueChange={([v]) => setNumSamples(v)} min={50} max={5000} step={50} />
        </div>
      </div>
      <div className="mb-3 text-sm text-[var(--text-muted)]">
        Integral of sin(pi*x) on [0,1] | Estimate: {lastEstimate.toFixed(4)} | True: {trueVal.toFixed(4)} | Error: {Math.abs(lastEstimate - trueVal).toFixed(4)}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <CanvasChart
          data={[
            { x: outsideX, y: outsideY, type: 'scatter', mode: 'markers', marker: { color: '#ef4444', size: 2, opacity: 0.4 }, name: 'Outside' },
            { x: insideX, y: insideY, type: 'scatter', mode: 'markers', marker: { color: '#10b981', size: 2, opacity: 0.4 }, name: 'Inside' },
            { x: curveX, y: curveY, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2.5 }, name: 'sin(pi*x)' },
          ]}
          layout={{
            height: 320,
            xaxis: { title: { text: 'x' }, range: [0, 1] },
            yaxis: { title: { text: 'y' }, range: [0, 1] },
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[
            { x: estimates.n, y: estimates.val, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2 }, name: 'Estimate' },
          ]}
          layout={{
            height: 320,
            xaxis: { title: { text: 'N samples' } },
            yaxis: { title: { text: 'Integral estimate' } },
            shapes: [
              { type: 'line', x0: 0, x1: numSamples, y0: trueVal, y1: trueVal, line: { color: '#ef4444', width: 1.5, dash: 'dash' } },
            ],
          }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
