"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function simulateGAN(dLR: number, gLR: number, collapseProb: number, seed: number) {
  const rng = mulberry32(seed);
  const steps = 200;
  const gLoss: number[] = [];
  const dLoss: number[] = [];
  const diversity: number[] = []; // 0-1: mode diversity

  let gBase = 2.0;
  let dBase = 0.8;
  let div = 0.3;
  let collapsed = false;

  for (let i = 0; i < steps; i++) {
    const noise = (rng() - 0.5) * 0.3;
    const dNoise = (rng() - 0.5) * 0.2;

    // Check for mode collapse
    if (!collapsed && rng() < collapseProb * 0.01) {
      collapsed = true;
    }

    if (collapsed) {
      // Mode collapse: G loss drops, D loss oscillates, diversity drops
      gBase = Math.max(0.5, gBase - gLR * 0.02);
      dBase = 0.7 + Math.sin(i * 0.15) * 0.3;
      div = Math.max(0.05, div - 0.005);
    } else {
      // Normal training: both losses oscillate around equilibrium
      const lrRatio = gLR / dLR;
      gBase = Math.max(0.3, gBase - gLR * 0.01 + noise * 0.1);
      dBase = Math.max(0.3, dBase + (lrRatio - 1) * 0.005 + dNoise * 0.05);
      // Oscillation
      gBase += Math.sin(i * 0.08) * 0.1 * lrRatio;
      dBase += Math.sin(i * 0.08 + Math.PI) * 0.05;
      div = Math.min(1, div + 0.003 + (rng() - 0.5) * 0.01);
    }

    gLoss.push(Math.max(0.1, gBase + noise * 0.15));
    dLoss.push(Math.max(0.1, dBase + dNoise * 0.1));
    diversity.push(Math.max(0, Math.min(1, div)));
  }

  return { gLoss, dLoss, diversity };
}

export default function GANTrainingDynamics({}: SimulationComponentProps) {
  const [dLR, setDLR] = useState(0.5);
  const [gLR, setGLR] = useState(0.5);
  const [collapseProb, setCollapseProb] = useState(0);

  const data = useMemo(() => simulateGAN(dLR, gLR, collapseProb, 42), [dLR, gLR, collapseProb]);
  const steps = Array.from({ length: 200 }, (_, i) => i);

  const lossTraces: any[] = [
    {
      type: 'scatter',
      x: steps,
      y: data.gLoss,
      mode: 'lines',
      line: { color: '#3b82f6', width: 2 },
      name: 'Generator loss',
    },
    {
      type: 'scatter',
      x: steps,
      y: data.dLoss,
      mode: 'lines',
      line: { color: '#ef4444', width: 2 },
      name: 'Discriminator loss',
    },
  ];

  const diversityTraces: any[] = [
    {
      type: 'scatter',
      x: steps,
      y: data.diversity,
      mode: 'lines',
      fill: 'tozeroy',
      line: { color: '#10b981', width: 2 },
      fillcolor: 'rgba(16, 185, 129, 0.15)',
      name: 'Mode diversity',
    },
  ];

  return (
    <SimulationPanel title="GAN Training Dynamics">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Discriminator LR factor: {dLR.toFixed(2)}
            </SimulationLabel>
            <Slider min={0.1} max={2} step={0.1} value={[dLR]} onValueChange={([v]) => setDLR(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Generator LR factor: {gLR.toFixed(2)}
            </SimulationLabel>
            <Slider min={0.1} max={2} step={0.1} value={[gLR]} onValueChange={([v]) => setGLR(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">
              Mode collapse risk: {collapseProb.toFixed(0)}%
            </SimulationLabel>
            <Slider min={0} max={100} step={5} value={[collapseProb]} onValueChange={([v]) => setCollapseProb(v)} className="w-full" />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasChart
            data={lossTraces}
            layout={{
              title: { text: 'Generator vs Discriminator Loss' },
              xaxis: { title: { text: 'Training step' } },
              yaxis: { title: { text: 'Loss' } },
              height: 350,
            }}
            style={{ width: '100%', height: 350 }}
          />
          <CanvasChart
            data={diversityTraces}
            layout={{
              title: { text: 'Output Diversity (Mode Coverage)' },
              xaxis: { title: { text: 'Training step' } },
              yaxis: { title: { text: 'Diversity' }, range: [0, 1.1] },
              height: 350,
            }}
            style={{ width: '100%', height: 350 }}
          />
        </div>
      </SimulationMain>

      <div className="mt-4 p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        {collapseProb > 30 ? (
          <p><strong className="text-red-400">Mode collapse detected:</strong> The generator has converged to producing only a few modes. Notice how diversity drops sharply while generator loss may appear low. The discriminator oscillates as it alternately detects and misses the collapsed modes.</p>
        ) : (
          <p>The adversarial training shows characteristic oscillation between generator and discriminator losses. Stable training occurs when neither network dominates. Adjust the learning rate ratio to see how imbalanced training affects convergence.</p>
        )}
      </div>
    </SimulationPanel>
  );
}
