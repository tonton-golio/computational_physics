"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function ProfileLikelihood({}: SimulationComponentProps) {
  const [nData, setNData] = useState(20);

  const { muVals, profileLL, mle, ci68, ci95 } = useMemo(() => {
    // Generate deterministic data
    const trueMu = 5.0;
    const trueSig = 2.0;
    const data: number[] = [];
    for (let i = 0; i < nData; i++) {
      data.push(trueMu + Math.sin(i * 2.71 + 0.5) * trueSig);
    }

    const mus: number[] = [];
    const pll: number[] = [];
    const muMin = 2;
    const muMax = 8;
    const steps = 200;

    let maxLL = -Infinity;
    let bestMu = trueMu;

    for (let i = 0; i <= steps; i++) {
      const mu = muMin + (muMax - muMin) * i / steps;
      mus.push(mu);
      // Profile over sigma: MLE sigma for given mu
      const ss = data.reduce((s, x) => s + (x - mu) ** 2, 0);
      const sigHat = Math.sqrt(ss / nData);
      // Log-likelihood at (mu, sigHat)
      let ll = 0;
      for (const x of data) {
        const z = (x - mu) / sigHat;
        ll += -0.5 * z * z - Math.log(sigHat) - 0.5 * Math.log(2 * Math.PI);
      }
      pll.push(ll);
      if (ll > maxLL) { maxLL = ll; bestMu = mu; }
    }

    // Convert to -2 * delta log-likelihood
    const deltaLL = pll.map((ll) => -2 * (ll - maxLL));

    // Find confidence intervals by linear interpolation
    const findCI = (threshold: number): [number, number] => {
      let lo = bestMu;
      let hi = bestMu;
      for (let i = 0; i < mus.length - 1; i++) {
        if (deltaLL[i] > threshold && deltaLL[i + 1] <= threshold) lo = mus[i + 1];
        if (deltaLL[i] <= threshold && deltaLL[i + 1] > threshold) hi = mus[i];
      }
      return [lo, hi];
    };

    return {
      muVals: mus,
      profileLL: deltaLL,
      mle: bestMu,
      ci68: findCI(1.0),
      ci95: findCI(3.84),
    };
  }, [nData]);

  return (
    <SimulationPanel title="Profile Likelihood">
      <SimulationConfig>
        <div>
          <SimulationLabel>Number of data points: {nData}</SimulationLabel>
          <Slider value={[nData]} onValueChange={([v]) => setNData(v)} min={5} max={100} step={1} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={[
          {
            x: muVals, y: profileLL, type: 'scatter', mode: 'lines',
            line: { color: '#3b82f6', width: 2.5 }, name: '-2 Delta ln L',
          },
          {
            x: [mle], y: [0], type: 'scatter', mode: 'markers',
            marker: { color: '#ef4444', size: 8 }, name: 'MLE',
          },
        ]}
        layout={{
          height: 400,
          xaxis: { title: { text: 'mu (parameter of interest)' } },
          yaxis: { title: { text: '-2 Delta ln L' }, range: [0, 10] },
          shapes: [
            { type: 'line', x0: muVals[0], x1: muVals[muVals.length - 1], y0: 1.0, y1: 1.0, line: { color: '#10b981', width: 1.5, dash: 'dash' } },
            { type: 'line', x0: muVals[0], x1: muVals[muVals.length - 1], y0: 3.84, y1: 3.84, line: { color: '#f59e0b', width: 1.5, dash: 'dash' } },
          ],
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          MLE: mu = {mle.toFixed(3)} | 68% CI: [{ci68[0].toFixed(2)}, {ci68[1].toFixed(2)}] | 95% CI: [{ci95[0].toFixed(2)}, {ci95[1].toFixed(2)}]
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
