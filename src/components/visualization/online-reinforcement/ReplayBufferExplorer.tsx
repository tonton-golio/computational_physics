"use client";

import { useState, useMemo, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationLabel, SimulationButton, SimulationSettings, SimulationConfig } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function simulate(bufferSize: number, batchSize: number, steps: number, seed: number) {
  const rng = mulberry32(seed);

  // Simulate loss with replay: sampling from buffer decorrelates data
  const lossReplay: number[] = [];
  const lossNoReplay: number[] = [];
  let replayAvg = 1.0, noReplayAvg = 1.0;

  // Track buffer fill level
  const fillLevel: number[] = [];
  let currentFill = 0;

  for (let t = 0; t < steps; t++) {
    currentFill = Math.min(currentFill + 1, bufferSize);
    fillLevel.push(currentFill / bufferSize);

    // Effective sample diversity: replay draws from full buffer, no-replay uses last batch
    const replayDiversity = Math.min(currentFill, bufferSize) / bufferSize;
    const noReplayNoise = 0.3 * Math.sin(t / 6) + 0.2 * Math.sin(t / 3.7);

    // Loss with replay: smooth decay, diversity reduces variance
    const replayGrad = -0.008 * replayAvg + 0.003 * (rng() - 0.5) * (1 - 0.7 * replayDiversity);
    replayAvg = Math.max(0.02, replayAvg + replayGrad);
    lossReplay.push(replayAvg);

    // Loss without replay: correlated data causes oscillation and slower convergence
    const noReplayGrad = -0.006 * noReplayAvg + noReplayNoise * 0.08 + 0.01 * (rng() - 0.5);
    noReplayAvg = Math.max(0.05, Math.min(1.5, noReplayAvg + noReplayGrad));
    lossNoReplay.push(noReplayAvg);
  }

  // Buffer composition: show which timesteps are in the buffer at the end
  const bufferStart = Math.max(0, steps - bufferSize);
  const bufferAge: number[] = [];
  const bufferIdx: number[] = [];
  for (let i = 0; i < Math.min(bufferSize, steps); i++) {
    bufferIdx.push(i);
    bufferAge.push(bufferStart + i);
  }

  // Sampled batch: random indices from the buffer
  const sampledSlots = new Set<number>();
  const batchRng = mulberry32(seed + 999);
  const actualBatch = Math.min(batchSize, bufferIdx.length);
  while (sampledSlots.size < actualBatch) {
    sampledSlots.add(Math.floor(batchRng() * bufferIdx.length));
  }

  return { lossReplay, lossNoReplay, fillLevel, bufferAge, bufferIdx, sampledSlots, actualBatch };
}

export default function ReplayBufferExplorer({}: SimulationComponentProps) {
  const [bufferSize, setBufferSize] = useState(200);
  const [batchSize, setBatchSize] = useState(32);
  const [steps, setSteps] = useState(300);
  const [seed, setSeed] = useState(17);

  const { lossTraces, bufferTraces } = useMemo(() => {
    const sim = simulate(bufferSize, batchSize, steps, seed);
    const x = Array.from({ length: steps }, (_, i) => i + 1);

    const lossTraces: ChartTrace[] = [
      { x, y: sim.lossReplay, type: 'scatter', mode: 'lines',
        name: 'With replay buffer', line: { color: '#4ade80', width: 2 } },
      { x, y: sim.lossNoReplay, type: 'scatter', mode: 'lines',
        name: 'Without replay (sequential)', line: { color: '#f87171', width: 2 } },
    ];

    // Buffer visualization: bar chart showing transition age, highlight sampled ones
    const numBars = Math.min(80, sim.bufferIdx.length);
    const stride = Math.max(1, Math.floor(sim.bufferIdx.length / numBars));
    const barX: number[] = [];
    const barY: number[] = [];
    const barColors: string[] = [];

    for (let i = 0; i < sim.bufferIdx.length; i += stride) {
      barX.push(i);
      barY.push(sim.bufferAge[i]);
      barColors.push(sim.sampledSlots.has(i) ? '#f59e0b' : '#3b82f6');
    }

    const bufferTraces: ChartTrace[] = [
      { x: barX, y: barY, type: 'bar',
        marker: { color: barColors }, name: 'Transition age (yellow = sampled)' },
    ];

    return { lossTraces, bufferTraces };
  }, [bufferSize, batchSize, steps, seed]);

  const handleResample = useCallback(() => setSeed((s) => s + 1), []);

  return (
    <SimulationPanel title="Experience Replay Buffer Explorer" caption="A circular buffer stores transitions (s, a, r, s'). Training samples random mini-batches, breaking temporal correlation and stabilizing learning.">
      <SimulationSettings>
        <SimulationButton variant="primary" onClick={handleResample}>
          Re-sample
        </SimulationButton>
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>Buffer size: {bufferSize}</SimulationLabel>
          <Slider value={[bufferSize]} onValueChange={([v]) => setBufferSize(v)} min={50} max={1000} step={50} />
        </div>
        <div>
          <SimulationLabel>Batch size: {batchSize}</SimulationLabel>
          <Slider value={[batchSize]} onValueChange={([v]) => setBatchSize(v)} min={8} max={128} step={8} />
        </div>
        <div>
          <SimulationLabel>Training steps: {steps}</SimulationLabel>
          <Slider value={[steps]} onValueChange={([v]) => setSteps(v)} min={100} max={600} step={25} />
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart data={lossTraces} layout={{
        title: { text: 'Training loss: replay vs sequential updates' },
        xaxis: { title: { text: 'training step' } },
        yaxis: { title: { text: 'loss' } },
        height: 340,
      }} style={{ width: '100%' }} />
      <div className="mt-4" />
      <CanvasChart data={bufferTraces} layout={{
        title: { text: 'Buffer contents (yellow bars = sampled for current batch)' },
        xaxis: { title: { text: 'buffer slot' } },
        yaxis: { title: { text: 'transition timestep' } },
        height: 260,
      }} style={{ width: '100%' }} />
      </SimulationMain>
    </SimulationPanel>
  );
}
