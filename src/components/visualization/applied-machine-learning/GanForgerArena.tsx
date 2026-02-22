"use client";

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationButton, SimulationPlayButton, SimulationCheckbox } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { mulberry32, gaussianPair, linspace } from './ml-utils';

interface TrainingState {
  epoch: number;
  // Generator parameters: means and stds for each mixture component
  genMeans: number[];
  genStds: number[];
  // Losses
  dLoss: number[];
  gLoss: number[];
  // Diversity metric
  diversity: number[];
}

const TARGET_MEANS = [-2, 1, 3.5];
const TARGET_STDS = [0.4, 0.5, 0.3];
const N_SAMPLES = 300;

function sampleTarget(rng: () => number, n: number): number[] {
  const samples: number[] = [];
  for (let i = 0; i < n; i++) {
    const comp = Math.floor(rng() * TARGET_MEANS.length);
    const [g] = gaussianPair(rng);
    samples.push(TARGET_MEANS[comp] + g * TARGET_STDS[comp]);
  }
  return samples;
}

function sampleGenerator(
  rng: () => number,
  means: number[],
  stds: number[],
  n: number,
  modeCollapse: boolean,
): number[] {
  const samples: number[] = [];
  for (let i = 0; i < n; i++) {
    const comp = modeCollapse ? 0 : Math.floor(rng() * means.length);
    const [g] = gaussianPair(rng);
    samples.push(means[comp] + g * stds[comp]);
  }
  return samples;
}

function histogram(data: number[], bins: number[], binWidth: number): number[] {
  const counts = new Array(bins.length).fill(0);
  for (const x of data) {
    const idx = Math.floor((x - bins[0]) / binWidth);
    if (idx >= 0 && idx < counts.length) counts[idx]++;
  }
  const total = data.length * binWidth;
  return counts.map((c) => c / Math.max(total, 1));
}

function diversityScore(samples: number[]): number {
  if (samples.length < 2) return 0;
  let sum = 0;
  const n = Math.min(samples.length, 100);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      sum += Math.abs(samples[i] - samples[j]);
    }
  }
  return (2 * sum) / (n * (n - 1));
}

function initState(): TrainingState {
  return {
    epoch: 0,
    genMeans: [0, 0, 0],
    genStds: [1.5, 1.5, 1.5],
    dLoss: [],
    gLoss: [],
    diversity: [],
  };
}

export default function GanForgerArena({}: SimulationComponentProps): React.ReactElement {
  const [state, setState] = useState<TrainingState>(initState);
  const [running, setRunning] = useState(false);
  const [modeCollapse, setModeCollapse] = useState(false);
  const [wasserstein, setWasserstein] = useState(false);
  const [speed, setSpeed] = useState(50);
  const _rafRef = useRef<number>(0);
  const runningRef = useRef(false);

  const trainStep = useCallback(
    (prev: TrainingState): TrainingState => {
      const rng = mulberry32(prev.epoch * 137 + 42);
      const lr = 0.02;

      // Current generator samples
      const genSamples = sampleGenerator(
        mulberry32(prev.epoch * 73 + 11),
        prev.genMeans,
        prev.genStds,
        N_SAMPLES,
        modeCollapse,
      );

      // Compute "discriminator loss" (how different gen is from target)
      const targetSamples = sampleTarget(mulberry32(prev.epoch * 53 + 7), N_SAMPLES);
      const bins = linspace(-5, 6, 50);
      const binWidth = bins[1] - bins[0];
      const targetHist = histogram(targetSamples, bins, binWidth);
      const genHist = histogram(genSamples, bins, binWidth);

      let dLoss = 0;
      let gLoss = 0;
      for (let i = 0; i < bins.length; i++) {
        if (wasserstein) {
          dLoss += Math.abs(targetHist[i] - genHist[i]) * binWidth;
        } else {
          const t = Math.max(targetHist[i], 1e-6);
          const g = Math.max(genHist[i], 1e-6);
          dLoss += -Math.log(t / (t + g)) * binWidth;
          gLoss += -Math.log(g / (t + g)) * binWidth;
        }
      }
      if (wasserstein) gLoss = dLoss;

      // Update generator: move means toward target, adjust stds
      const newMeans = prev.genMeans.map((m, i) => {
        const target = TARGET_MEANS[i];
        const noise = (rng() - 0.5) * 0.1;
        return m + lr * (target - m) + noise;
      });

      const newStds = prev.genStds.map((s, i) => {
        const target = TARGET_STDS[i];
        const noise = (rng() - 0.5) * 0.02;
        return Math.max(0.1, s + lr * (target - s) + noise);
      });

      const div = diversityScore(genSamples);

      return {
        epoch: prev.epoch + 1,
        genMeans: newMeans,
        genStds: newStds,
        dLoss: [...prev.dLoss, dLoss],
        gLoss: [...prev.gLoss, gLoss],
        diversity: [...prev.diversity, div],
      };
    },
    [modeCollapse, wasserstein],
  );

  // Training loop
  useEffect(() => {
    if (!running) return;
    runningRef.current = true;

    const interval = setInterval(() => {
      if (!runningRef.current) return;
      setState((prev) => {
        if (prev.epoch >= 200) {
          setRunning(false);
          return prev;
        }
        return trainStep(prev);
      });
    }, Math.max(10, 200 - speed * 2));

    return () => {
      runningRef.current = false;
      clearInterval(interval);
    };
  }, [running, speed, trainStep]);

  const trainN = useCallback(
    (n: number) => {
      setState((prev) => {
        let s = prev;
        for (let i = 0; i < n && s.epoch < 200; i++) {
          s = trainStep(s);
        }
        return s;
      });
    },
    [trainStep],
  );

  // Generate current samples for display
  const targetSamples = sampleTarget(mulberry32(9999), N_SAMPLES);
  const genSamples = sampleGenerator(
    mulberry32(state.epoch * 73 + 11),
    state.genMeans,
    state.genStds,
    N_SAMPLES,
    modeCollapse,
  );

  const bins = linspace(-5, 6, 50);
  const binWidth = bins[1] - bins[0];
  const targetHist = histogram(targetSamples, bins, binWidth);
  const genHist = histogram(genSamples, bins, binWidth);

  const currentDiversity = state.diversity.length > 0
    ? state.diversity[state.diversity.length - 1]
    : diversityScore(genSamples);

  return (
    <SimulationPanel title="GAN Forger Arena">
      <SimulationSettings>
        <div className="flex flex-wrap gap-2">
          <SimulationButton variant="primary" onClick={() => trainN(5)} disabled={running || state.epoch >= 200}>
            Train D 5 steps
          </SimulationButton>
          <SimulationButton variant="primary" onClick={() => trainN(5)} disabled={running || state.epoch >= 200}>
            Train G 5 steps
          </SimulationButton>
          <SimulationPlayButton
            isRunning={running}
            onToggle={() => setRunning((r) => !r)}
            labels={{ play: 'Full Training', pause: 'Pause' }}
            disabled={state.epoch >= 200}
          />
          <SimulationButton variant="secondary" onClick={() => setState(initState())}>
            Reset
          </SimulationButton>
        </div>
        <div className="flex flex-col gap-1">
          <SimulationCheckbox checked={wasserstein} onChange={setWasserstein} label="Wasserstein loss" />
          <SimulationCheckbox checked={modeCollapse} onChange={setModeCollapse} label="Force mode collapse" />
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>Speed: {speed}</SimulationLabel>
          <Slider min={10} max={100} step={5} value={[speed]} onValueChange={([v]) => setSpeed(v)} />
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Distribution comparison */}
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">
            Target vs Generated Distribution
          </p>
          <CanvasChart
            data={[
              {
                x: bins,
                y: targetHist,
                type: 'scatter',
                mode: 'lines',
                name: 'Target (real)',
                line: { color: '#3b82f6', width: 2 },
                fill: 'tozeroy',
              },
              {
                x: bins,
                y: genHist,
                type: 'scatter',
                mode: 'lines',
                name: 'Generator (fake)',
                line: { color: '#ef4444', width: 2 },
                fill: 'tozeroy',
              },
            ]}
            layout={{
              xaxis: { title: { text: 'Value' } },
              yaxis: { title: { text: 'Density' } },
              margin: { t: 20, r: 20, b: 45, l: 55 },
            }}
            style={{ width: '100%', height: 320 }}
          />
        </div>

        {/* Loss curves */}
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Training Losses</p>
          {state.dLoss.length > 0 ? (
            <CanvasChart
              data={[
                {
                  x: Array.from({ length: state.dLoss.length }, (_, i) => i),
                  y: state.dLoss,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Discriminator',
                  line: { color: '#3b82f6', width: 2 },
                },
                {
                  x: Array.from({ length: state.gLoss.length }, (_, i) => i),
                  y: state.gLoss,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Generator',
                  line: { color: '#ef4444', width: 2 },
                },
              ]}
              layout={{
                xaxis: { title: { text: 'Epoch' } },
                yaxis: { title: { text: 'Loss' } },
                margin: { t: 20, r: 20, b: 45, l: 55 },
              }}
              style={{ width: '100%', height: 320 }}
            />
          ) : (
            <div className="flex h-[320px] items-center justify-center text-sm text-[var(--text-muted)]">
              Start training to see loss curves
            </div>
          )}
        </div>
        </div>

        {modeCollapse && currentDiversity < 1.5 && state.epoch > 10 && (
          <div className="mt-3 rounded bg-red-900/30 p-3 text-sm text-red-300">
            Mode collapse detected. The generator is only producing samples from
            one mode of the target distribution. Diversity score dropped to{' '}
            {currentDiversity.toFixed(2)}. In real GANs, this happens when the
            generator finds one pattern that fools the discriminator and keeps repeating it.
          </div>
        )}

        {!modeCollapse && state.epoch >= 150 && currentDiversity > 2.0 && (
          <div className="mt-3 rounded bg-emerald-900/30 p-3 text-sm text-emerald-300">
            Near equilibrium. The generator has learned to match all three modes of the
            target distribution. Both networks have become experts at their respective
            tasks.
          </div>
        )}
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          Epoch: {state.epoch}/200
          <br />
          Diversity: {currentDiversity.toFixed(2)}
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
