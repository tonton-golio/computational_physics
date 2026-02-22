"use client";

import { useState, useMemo } from 'react';
import { gaussianPair, gaussPdf } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

export default function EntropyDemo({}: SimulationComponentProps) {
  const [loc, setLoc] = useState(-8);
  const [scale, setScale] = useState(3.0);
  const [sizeExp, setSizeExp] = useState(8);
  const size = Math.pow(2, sizeExp);

  const result = useMemo(() => {
    const rng = seededRandom(42);

    // Discrete: sample from normal, round to integers
    const xDiscrete: number[] = [];
    for (let i = 0; i < size; i++) {
      xDiscrete.push(Math.round(loc + scale * gaussianPair(rng)[0]));
    }

    // Compute discrete entropy
    const counts: Record<number, number> = {};
    for (const v of xDiscrete) {
      counts[v] = (counts[v] || 0) + 1;
    }
    let H_discrete = 0;
    for (const key of Object.keys(counts)) {
      const p = counts[Number(key)] / xDiscrete.length;
      if (p > 0) {
        H_discrete -= p * Math.log2(p);
      }
    }

    // Continuous: evaluate Gaussian PDF on linspace and compute differential entropy
    const nPts = 500;
    const xCont: number[] = [];
    const fxCont: number[] = [];
    const xMin = -40;
    const xMax = 40;
    const dx = (xMax - xMin) / (nPts - 1);
    for (let i = 0; i < nPts; i++) {
      const x = xMin + i * dx;
      xCont.push(x);
      fxCont.push(gaussPdf(x, loc, scale));
    }

    let H_continuous = 0;
    for (let i = 0; i < nPts; i++) {
      if (fxCont[i] > 1e-15) {
        H_continuous -= fxCont[i] * Math.log2(fxCont[i]) * dx;
      }
    }

    // Build histogram bins for discrete
    const minVal = Math.min(...xDiscrete);
    const maxVal = Math.max(...xDiscrete);
    const bins: number[] = [];
    const binCounts: number[] = [];
    for (let v = minVal; v <= maxVal; v++) {
      bins.push(v);
      binCounts.push(counts[v] || 0);
    }

    return {
      bins,
      binCounts,
      H_discrete,
      xCont,
      fxCont,
      H_continuous,
    };
  }, [loc, scale, size]);

  return (
    <SimulationPanel title="Discrete vs Continuous Entropy">
      <SimulationConfig>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-sm">Location (mean): {loc}</SimulationLabel>
            <Slider
              min={-20} max={4} step={1} value={[loc]}
              onValueChange={([v]) => setLoc(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-sm">Scale (std): {scale.toFixed(1)}</SimulationLabel>
            <Slider
              min={0.5} max={10} step={0.1} value={[scale]}
              onValueChange={([v]) => setScale(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel className="text-[var(--text-muted)] text-sm">Sample size: 2^{sizeExp} = {size}</SimulationLabel>
            <Slider
              min={4} max={10} step={1} value={[sizeExp]}
              onValueChange={([v]) => setSizeExp(v)}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <CanvasChart
            data={[{
              x: result.bins,
              y: result.binCounts,
              type: 'bar' as const,
              marker: { color: '#ef4444' },
              opacity: 0.8,
              name: 'Discrete samples',
            }]}
            layout={{
              title: { text: `Discrete Entropy: H_s = ${result.H_discrete.toFixed(3)} bits` },
              xaxis: { title: { text: 'Value' }, range: [-40, 40] },
              yaxis: { title: { text: 'Count' } },
              height: 320,
              margin: { t: 40, b: 50, l: 50, r: 20 },
            }}
            style={{ width: '100%' }}
          />
        </div>
        <div>
          <CanvasChart
            data={[{
              x: result.xCont,
              y: result.fxCont,
              type: 'scatter' as const,
              mode: 'lines' as const,
              fill: 'tozeroy',
              fillcolor: 'rgba(147,51,234,0.5)',
              line: { color: 'rgba(147,51,234,1)' },
              name: 'Gaussian PDF',
            }]}
            layout={{
              title: { text: `Differential Entropy: H_d = ${result.H_continuous.toFixed(3)} bits` },
              xaxis: { title: { text: 'Value' }, range: [-40, 40] },
              yaxis: { title: { text: 'f(x)' } },
              height: 320,
              margin: { t: 40, b: 50, l: 50, r: 20 },
            }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
      <p className="text-[var(--text-soft)] text-xs mt-3">
        Discrete Shannon entropy is always non-negative. Differential entropy can be negative and is not a direct measure of information content. Note that differential entropy is translation-invariant.
      </p>
      </SimulationMain>
    </SimulationPanel>
  );
}
