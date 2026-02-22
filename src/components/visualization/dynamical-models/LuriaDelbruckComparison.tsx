"use client";

import { useState, useMemo } from 'react';
import { mulberry32 } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/* ── Seeded pseudo-random number generator ─────────────────────────────── */
function createRng(seed: number) {
  return mulberry32((seed + 1) * 2654435761);
}

/* ── Draw a Poisson sample using the inversion method ────────────────── */
function poissonSample(lambda: number, rand: () => number): number {
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1.0;
  do {
    k++;
    p *= rand();
  } while (p > L);
  return k - 1;
}

/* ── Build a histogram from raw samples ──────────────────────────────── */
function buildHistogram(samples: number[], maxBin: number) {
  const counts = new Array(maxBin + 1).fill(0);
  for (const s of samples) {
    const bin = Math.min(Math.max(0, Math.round(s)), maxBin);
    counts[bin]++;
  }
  return counts;
}

/* ── Statistics helpers ───────────────────────────────────────────────── */
function computeStats(samples: number[]) {
  const n = samples.length;
  if (n === 0) return { mean: 0, variance: 0, fano: 0 };
  const mean = samples.reduce((a, b) => a + b, 0) / n;
  const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const fano = mean > 0 ? variance / mean : 0;
  return { mean, variance, fano };
}

export default function LuriaDelbruckComparison({}: SimulationComponentProps) {
  const [nCultures, setNCultures] = useState(50);
  const [mu, setMu] = useState(1.0);

  const { directed, spontaneous, maxBin } = useMemo(() => {
    const rand = createRng(42);

    /* ── Directed mutation model (Poisson): each culture independently
         acquires mutations with fixed probability → Poisson(μ) ───────── */
    const directedSamples: number[] = [];
    for (let i = 0; i < nCultures; i++) {
      directedSamples.push(poissonSample(mu, rand));
    }

    /* ── Spontaneous mutation model (Luria-Delbrück): mutations arise
         randomly during exponential growth. Early mutations produce large
         clones ("jackpots"), late mutations produce few descendants.
         Simulation: draw nMutations ~ Poisson(μ), then for each mutation
         event draw clone size ~ Geometric(p=0.5). Total mutants per
         culture = sum of clone sizes. ────────────────────────────────── */
    const spontaneousSamples: number[] = [];
    for (let i = 0; i < nCultures; i++) {
      const nMutations = poissonSample(mu, rand);
      let total = 0;
      for (let j = 0; j < nMutations; j++) {
        // Geometric random variable with p = 0.5: E[X] = 1/p = 2
        // P(X = k) = (1-p)^(k-1) * p, k = 1,2,3,...
        let cloneSize = 1;
        while (rand() > 0.5) {
          cloneSize++;
        }
        total += cloneSize;
      }
      spontaneousSamples.push(total);
    }

    const allMax = Math.max(
      ...directedSamples,
      ...spontaneousSamples,
      5,
    );
    const maxBin = Math.min(allMax + 2, 80);

    const directedCounts = buildHistogram(directedSamples, maxBin);
    const spontaneousCounts = buildHistogram(spontaneousSamples, maxBin);

    const directedStats = computeStats(directedSamples);
    const spontaneousStats = computeStats(spontaneousSamples);

    // Identify jackpot threshold for the spontaneous model: mean + 3*sigma
    const jackpotThreshold = spontaneousStats.mean + 3 * Math.sqrt(spontaneousStats.variance);

    return {
      directed: {
        counts: directedCounts,
        stats: directedStats,
        samples: directedSamples,
      },
      spontaneous: {
        counts: spontaneousCounts,
        stats: spontaneousStats,
        samples: spontaneousSamples,
        jackpotThreshold,
      },
      maxBin,
    };
  }, [nCultures, mu]);

  /* ── Build bar chart data ──────────────────────────────────────────── */
  const ks = Array.from({ length: maxBin + 1 }, (_, i) => i);

  // Directed model: single blue bar trace
  const directedTrace = {
    x: ks,
    y: directed.counts,
    type: 'bar' as const,
    marker: { color: '#3b82f6' },
    name: 'Directed (Poisson)',
    opacity: 0.8,
  };

  // Spontaneous model: split into normal bars and jackpot bars
  const threshold = spontaneous.jackpotThreshold;
  const normalCounts = spontaneous.counts.map((c, i) => (i <= threshold ? c : 0));
  const jackpotCounts = spontaneous.counts.map((c, i) => (i > threshold ? c : 0));

  const spontNormalTrace = {
    x: ks,
    y: normalCounts,
    type: 'bar' as const,
    marker: { color: '#f97316' },
    name: 'Spontaneous (LD)',
    opacity: 0.8,
  };

  const spontJackpotTrace = {
    x: ks,
    y: jackpotCounts,
    type: 'bar' as const,
    marker: { color: '#ef4444' },
    name: 'Jackpot cultures',
    opacity: 0.9,
  };

  const chartHeight = 380;
  const commonMargin = { t: 40, b: 50, l: 55, r: 20 };

  return (
    <SimulationPanel title="Luria-Delbr&uuml;ck: Directed vs Spontaneous Mutation" caption="Compare the mutant count distribution across parallel cultures under two competing hypotheses: directed mutation (Poisson) versus spontaneous mutation during growth (Luria-Delbr&uuml;ck).">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Number of cultures: {nCultures}
          </SimulationLabel>
          <Slider
            value={[nCultures]}
            onValueChange={([v]) => setNCultures(v)}
            min={10}
            max={200}
            step={10}
          />
        </div>
        <div>
          <SimulationLabel>
            Mutation rate &mu;: {mu.toFixed(1)}
          </SimulationLabel>
          <Slider
            value={[mu]}
            onValueChange={([v]) => setMu(v)}
            min={0.1}
            max={5.0}
            step={0.1}
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <CanvasChart
            data={[directedTrace] as any}
            layout={{
              title: { text: 'Directed Mutation (Poisson)' },
              margin: commonMargin,
              xaxis: { title: { text: 'Mutant count per culture' } },
              yaxis: { title: { text: 'Frequency' } },
              bargap: 0.05,
              height: chartHeight,
            }}
            style={{ width: '100%', height: chartHeight }}
          />
          <div className="mt-2 grid grid-cols-3 gap-2 text-sm">
            <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
              <div className="text-[var(--text-soft)]">Mean</div>
              <div className="text-[var(--text-strong)] font-mono">
                {directed.stats.mean.toFixed(2)}
              </div>
            </div>
            <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
              <div className="text-[var(--text-soft)]">Variance</div>
              <div className="text-[var(--text-strong)] font-mono">
                {directed.stats.variance.toFixed(2)}
              </div>
            </div>
            <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
              <div className="text-[var(--text-soft)]">Fano factor</div>
              <div className="text-[var(--text-strong)] font-mono">
                {directed.stats.fano.toFixed(2)}
              </div>
            </div>
          </div>
        </div>

        <div>
          <CanvasChart
            data={[spontNormalTrace, spontJackpotTrace] as any}
            layout={{
              title: { text: 'Spontaneous Mutation (Luria-Delbrück)' },
              margin: commonMargin,
              xaxis: { title: { text: 'Mutant count per culture' } },
              yaxis: { title: { text: 'Frequency' } },
              barmode: 'stack',
              bargap: 0.05,
              height: chartHeight,
            }}
            style={{ width: '100%', height: chartHeight }}
          />
          <div className="mt-2 grid grid-cols-3 gap-2 text-sm">
            <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
              <div className="text-[var(--text-soft)]">Mean</div>
              <div className="text-[var(--text-strong)] font-mono">
                {spontaneous.stats.mean.toFixed(2)}
              </div>
            </div>
            <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
              <div className="text-[var(--text-soft)]">Variance</div>
              <div className="text-[var(--text-strong)] font-mono">
                {spontaneous.stats.variance.toFixed(2)}
              </div>
            </div>
            <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
              <div className="text-[var(--text-soft)]">Fano factor</div>
              <div className="text-[var(--text-strong)] font-mono">
                {spontaneous.stats.fano.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      </div>
      </SimulationMain>

      {/* ── Explanation ────────────────────────────────────────────────── */}
      <div className="mt-3 text-sm text-[var(--text-muted)] space-y-2">
        <p>
          <strong className="text-[var(--text-muted)]">Directed mutation hypothesis:</strong>{' '}
          Mutations are induced by the selective agent. Every cell has the same small
          probability of mutating upon exposure, so the number of mutants per culture
          follows a Poisson distribution with Fano factor (Variance / Mean) close to 1.
        </p>
        <p>
          <strong className="text-[var(--text-muted)]">Spontaneous mutation hypothesis (Luria-Delbr&uuml;ck):</strong>{' '}
          Mutations arise randomly during growth, before selection. An early mutation
          produces a large clone of resistant descendants (&ldquo;jackpot&rdquo;), while
          a late mutation yields only a few. This leads to a highly overdispersed
          distribution with Fano factor much greater than 1 and a characteristic
          heavy right tail. The red bars mark jackpot cultures beyond
          3 standard deviations from the mean.
        </p>
        <p>
          Luria and Delbr&uuml;ck&apos;s 1943 fluctuation test demonstrated that the
          observed variance in bacterial resistance counts vastly exceeded the Poisson
          prediction, confirming that mutations are spontaneous and not directed.
        </p>
      </div>
    </SimulationPanel>
  );
}
