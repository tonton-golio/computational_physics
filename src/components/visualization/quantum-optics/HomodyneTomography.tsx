"use client";

import { useState, useMemo } from 'react';
import { gaussianPair } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Homodyne Tomography
 *
 * Simulates quadrature measurements at many phase angles and reconstructs
 * the Wigner function via inverse Radon back-projection. Compares the
 * reconstruction against the true Wigner function.
 *
 * States: coherent |alpha=2> and cat state (|alpha> + |-alpha>)/sqrt(N).
 */

type StateChoice = 'coherent' | 'cat';

const STATE_OPTIONS = [
  { label: 'Coherent', value: 'coherent' as const },
  { label: 'Cat', value: 'cat' as const },
];

// Seeded PRNG (xorshift32) for deterministic sampling
function xorshift32(seed: number) {
  let state = seed | 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}

// True Wigner function
function trueWigner(
  state: StateChoice,
  x: number,
  p: number,
  alpha: number,
): number {
  if (state === 'coherent') {
    const x0 = Math.sqrt(2) * alpha;
    return (1 / Math.PI) * Math.exp(-((x - x0) ** 2) - p * p);
  }
  // Cat state: (|alpha> + |-alpha>) / sqrt(N)
  const x0 = Math.sqrt(2) * alpha;
  const norm = 2 * (1 + Math.exp(-2 * alpha * alpha));
  const g1 = Math.exp(-((x - x0) ** 2) - p * p);
  const g2 = Math.exp(-((x + x0) ** 2) - p * p);
  const interference = 2 * Math.cos(2 * p * x0) * Math.exp(-x * x - p * p);
  return (1 / Math.PI) * (g1 + g2 + interference) / norm * 2;
}

// Quadrature mean and variance for sampling
function quadratureMoments(
  state: StateChoice,
  theta: number,
  alpha: number,
): { mean: number; std: number } {
  if (state === 'coherent') {
    const mean = Math.sqrt(2) * alpha * Math.cos(theta);
    return { mean, std: 0.5 };
  }
  // Cat state: mixture of two Gaussians — effective mean is 0 (symmetric cat), std broadened
  const variance = 0.25 + 2 * alpha * alpha * Math.cos(theta) ** 2;
  return { mean: 0, std: Math.sqrt(variance) };
}

// Sample quadrature measurements for a given phase angle
function sampleQuadrature(
  state: StateChoice,
  theta: number,
  alpha: number,
  nSamples: number,
  rng: () => number,
): number[] {
  const samples: number[] = [];

  if (state === 'coherent') {
    const { mean } = quadratureMoments(state, theta, alpha);
    for (let i = 0; i < nSamples; i++) {
      samples.push(mean + 0.5 * gaussianPair(rng)[0]);
    }
  } else {
    // Cat: sample from mixture of two Gaussians
    const proj = Math.sqrt(2) * alpha * Math.cos(theta);
    for (let i = 0; i < nSamples; i++) {
      const component = rng() < 0.5 ? 1 : -1;
      samples.push(component * proj + 0.5 * gaussianPair(rng)[0]);
    }
  }

  return samples;
}

// Inverse Radon back-projection on a grid
function backproject(
  allSamples: { theta: number; values: number[] }[],
  gridSize: number,
  range: number,
): { grid: number[][]; xvec: number[]; pvec: number[] } {
  const xvec = Array.from({ length: gridSize }, (_, i) =>
    -range + (2 * range * i) / (gridSize - 1),
  );
  const pvec = Array.from({ length: gridSize }, (_, i) =>
    -range + (2 * range * i) / (gridSize - 1),
  );

  // Accumulate back-projection
  const grid: number[][] = Array.from({ length: gridSize }, () =>
    new Array(gridSize).fill(0),
  );

  const binWidth = (2 * range) / 40;

  for (const { theta, values } of allSamples) {
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);

    // Build histogram of quadrature values
    const nBins = 40;
    const hist = new Array(nBins).fill(0);
    for (const v of values) {
      const bin = Math.floor(((v + range) / (2 * range)) * nBins);
      if (bin >= 0 && bin < nBins) hist[bin]++;
    }

    // Normalize histogram to density
    const totalSamples = values.length;
    for (let b = 0; b < nBins; b++) {
      hist[b] /= totalSamples * binWidth;
    }

    // Back-project: for each grid point, find its projection onto the theta direction
    for (let j = 0; j < gridSize; j++) {
      for (let i = 0; i < gridSize; i++) {
        const proj = xvec[i] * cosT + pvec[j] * sinT;
        const bin = Math.floor(((proj + range) / (2 * range)) * nBins);
        if (bin >= 0 && bin < nBins) {
          grid[j][i] += hist[bin];
        }
      }
    }
  }

  // Normalize by number of angles
  const nAngles = allSamples.length;
  for (let j = 0; j < gridSize; j++) {
    for (let i = 0; i < gridSize; i++) {
      grid[j][i] = (grid[j][i] * Math.PI) / nAngles - 1 / (2 * Math.PI);
    }
  }

  return { grid, xvec, pvec };
}

// Compute fidelity estimate as overlap between normalised grids
function estimateFidelity(
  reconstructed: number[][],
  trueGrid: number[][],
  gridSize: number,
): number {
  let sumRec2 = 0, sumTrue2 = 0, sumCross = 0;
  for (let j = 0; j < gridSize; j++) {
    for (let i = 0; i < gridSize; i++) {
      sumRec2 += reconstructed[j][i] ** 2;
      sumTrue2 += trueGrid[j][i] ** 2;
      sumCross += reconstructed[j][i] * trueGrid[j][i];
    }
  }
  const normRec = Math.sqrt(sumRec2);
  const normTrue = Math.sqrt(sumTrue2);
  if (normRec < 1e-12 || normTrue < 1e-12) return 0;
  return Math.max(0, Math.min(1, sumCross / (normRec * normTrue)));
}

export default function HomodyneTomography({}: SimulationComponentProps) {
  const [nSamples, setNSamples] = useState(500);
  const [stateChoice, setStateChoice] = useState<StateChoice>('coherent');

  const alpha = 2.0;
  const gridSize = 60;
  const range = 5;
  const nAngles = 32;

  const { reconstructedGrid, trueGrid, xvec, pvec, fidelity } = useMemo(() => {
    const rng = xorshift32(42);
    const angles = Array.from({ length: nAngles }, (_, i) =>
      (Math.PI * i) / nAngles,
    );

    const samplesPerAngle = Math.max(1, Math.round(nSamples / nAngles));

    const allSamples = angles.map((theta) => ({
      theta,
      values: sampleQuadrature(stateChoice, theta, alpha, samplesPerAngle, rng),
    }));

    const { grid: reconstructedGrid, xvec, pvec } = backproject(
      allSamples,
      gridSize,
      range,
    );

    // True Wigner on same grid
    const trueGrid: number[][] = [];
    for (let j = 0; j < gridSize; j++) {
      const row: number[] = [];
      for (let i = 0; i < gridSize; i++) {
        row.push(trueWigner(stateChoice, xvec[i], pvec[j], alpha));
      }
      trueGrid.push(row);
    }

    const fidelity = estimateFidelity(reconstructedGrid, trueGrid, gridSize);

    return { reconstructedGrid, trueGrid, xvec, pvec, fidelity };
  }, [nSamples, stateChoice]);

  return (
    <SimulationPanel title="Homodyne Tomography" caption={`Quadrature measurements at ${nAngles} phase angles are back-projected to reconstruct the Wigner function. Increase samples to see the reconstruction converge to the true state.`}>
      <SimulationSettings>
        <div>
          <SimulationToggle
            options={STATE_OPTIONS}
            value={stateChoice}
            onChange={(v) => setStateChoice(v as StateChoice)}
          />
        </div>
      </SimulationSettings>

      <SimulationConfig>
        <div>
          <SimulationLabel>Samples: {nSamples}</SimulationLabel>
          <Slider
            value={[nSamples]}
            onValueChange={(v) => setNSamples(Math.round(v[0]))}
            min={10}
            max={5000}
            step={10}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
        {/* Side-by-side heatmaps */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasHeatmap
            data={[{
              z: reconstructedGrid,
              x: xvec,
              y: pvec,
              type: 'heatmap',
              colorscale: 'RdBu',
              colorbar: { title: { text: 'W(q,p)' } },
            }]}
            layout={{
              title: { text: 'Reconstructed Wigner' },
              xaxis: { title: { text: 'q' } },
              yaxis: { title: { text: 'p' }, scaleanchor: 'x' },
              height: 400,
              margin: { t: 35, r: 20, b: 45, l: 50 },
            }}
            style={{ width: '100%', height: '400px' }}
          />
          <CanvasHeatmap
            data={[{
              z: trueGrid,
              x: xvec,
              y: pvec,
              type: 'heatmap',
              colorscale: 'RdBu',
              colorbar: { title: { text: 'W(q,p)' } },
            }]}
            layout={{
              title: { text: 'True Wigner' },
              xaxis: { title: { text: 'q' } },
              yaxis: { title: { text: 'p' }, scaleanchor: 'x' },
              height: 400,
              margin: { t: 35, r: 20, b: 45, l: 50 },
            }}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
      </SimulationMain>

      <SimulationResults>
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Total samples</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {nSamples}
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Reconstruction fidelity</div>
            <div className="text-base font-mono font-semibold text-[var(--text-strong)]">
              {(fidelity * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
