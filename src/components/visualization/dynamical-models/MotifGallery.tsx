'use client';

import React, { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';

/**
 * MotifGallery: Interactive gallery of network motifs.
 *
 * Four motifs:
 * 1. Negative autoregulation — dX/dt = beta/(1+(X/K)^n) - gamma*X
 * 2. Positive feedback — dX/dt = beta*(X/K)^n/(1+(X/K)^n) + beta0 - gamma*X
 * 3. Coherent feed-forward loop — X activates Y and Z, Y activates Z (AND gate)
 * 4. Toggle switch — dU/dt = alpha/(1+V^n) - U, dV/dt = alpha/(1+U^n) - V
 *
 * Each motif shows its step response: what happens when a signal turns on at t=0.
 */

type MotifId = 'negative-autoregulation' | 'positive-feedback' | 'feed-forward-loop' | 'toggle-switch';

interface MotifInfo {
  id: MotifId;
  title: string;
  shortName: string;
  personality: string;
  description: string;
  color: string;
}

const MOTIFS: MotifInfo[] = [
  {
    id: 'negative-autoregulation',
    title: 'Negative Autoregulation',
    shortName: 'NAR',
    personality: 'The Thermostat',
    description: 'Gene X represses itself. Reaches steady state fast, reduces noise, and is robust to parameter changes.',
    color: '#3b82f6',
  },
  {
    id: 'positive-feedback',
    title: 'Positive Feedback',
    shortName: 'PF',
    personality: 'The Commitment Device',
    description: 'Gene X activates itself. Creates bistability and memory. Once the cell commits to the high state, it stays there.',
    color: '#22c55e',
  },
  {
    id: 'feed-forward-loop',
    title: 'Coherent Feed-Forward Loop',
    shortName: 'FFL',
    personality: 'The Delay Timer',
    description: 'X activates Z directly and through Y. Z requires both X and Y (AND gate). Brief signals are filtered out; only sustained signals pass.',
    color: '#f97316',
  },
  {
    id: 'toggle-switch',
    title: 'Toggle Switch',
    shortName: 'TS',
    personality: 'The Memory Bank',
    description: 'Genes U and V mutually repress each other. One wins and stays dominant. A pulse of inducer flips the switch.',
    color: '#a855f7',
  },
];

function simulateNAR(tMax: number, dt: number): { t: number[]; x: number[]; xNoFB: number[] } {
  const beta = 10;
  const K = 1;
  const n = 4;
  const gamma = 1;

  const nSteps = Math.round(tMax / dt);
  const t: number[] = [];
  const x: number[] = [];
  const xNoFB: number[] = [];

  let xCurr = 0;
  let xNoFBCurr = 0;

  // Effective beta for no-feedback comparison: same steady state
  // NAR steady state: solve beta/(1+(X/K)^n) = gamma*X numerically
  // For simplicity, compute it
  const xSS = (() => {
    let xg = 0;
    for (let i = 0; i < 1000; i++) {
      xg = beta / (gamma * (1 + Math.pow(xg / K, n)));
      if (xg < 0) xg = 0;
    }
    return xg;
  })();
  const betaEff = gamma * xSS;

  for (let i = 0; i <= nSteps; i++) {
    t.push(i * dt);
    x.push(xCurr);
    xNoFB.push(xNoFBCurr);

    // Step signal: production on after t=0 (always on in this sim)
    const dxNAR = (beta / (1 + Math.pow(xCurr / K, n)) - gamma * xCurr) * dt;
    const dxNoFB = (betaEff - gamma * xNoFBCurr) * dt;

    xCurr = Math.max(0, xCurr + dxNAR);
    xNoFBCurr = Math.max(0, xNoFBCurr + dxNoFB);
  }

  return { t, x, xNoFB };
}

function simulatePF(tMax: number, dt: number): { t: number[]; xLow: number[]; xHigh: number[] } {
  const beta = 5;
  const K = 1;
  const n = 4;
  const beta0 = 0.1;
  const gamma = 1;

  const nSteps = Math.round(tMax / dt);
  const t: number[] = [];
  const xLow: number[] = [];
  const xHigh: number[] = [];

  let xL = 0.01; // Start near low state
  let xH = 4.0;  // Start near high state

  for (let i = 0; i <= nSteps; i++) {
    t.push(i * dt);
    xLow.push(xL);
    xHigh.push(xH);

    const fL = beta * Math.pow(xL / K, n) / (1 + Math.pow(xL / K, n)) + beta0 - gamma * xL;
    const fH = beta * Math.pow(xH / K, n) / (1 + Math.pow(xH / K, n)) + beta0 - gamma * xH;

    xL = Math.max(0, xL + fL * dt);
    xH = Math.max(0, xH + fH * dt);
  }

  return { t, xLow, xHigh };
}

function simulateFFL(tMax: number, dt: number): { t: number[]; signal: number[]; y: number[]; z: number[] } {
  // X is a step signal
  // Y: dY/dt = betaY * X - gammaY * Y
  // Z: dZ/dt = betaZ * min(X, Y/Kyz) - gammaZ * Z  (AND gate)
  const betaY = 2;
  const gammaY = 0.5;
  const betaZ = 2;
  const gammaZ = 0.5;
  const Kyz = 1;

  const nSteps = Math.round(tMax / dt);
  const t: number[] = [];
  const signal: number[] = [];
  const y: number[] = [];
  const z: number[] = [];

  let yCurr = 0;
  let zCurr = 0;

  for (let i = 0; i <= nSteps; i++) {
    const time = i * dt;
    // Step signal: on from t=2 to t=tMax-2
    const X = (time >= 2 && time <= tMax - 2) ? 1 : 0;

    t.push(time);
    signal.push(X * 3); // Scale for visualization
    y.push(yCurr);
    z.push(zCurr);

    const dy = (betaY * X - gammaY * yCurr) * dt;
    // AND gate: Z needs both X and sufficient Y
    const andSignal = X * (yCurr / (Kyz + yCurr));
    const dz = (betaZ * andSignal - gammaZ * zCurr) * dt;

    yCurr = Math.max(0, yCurr + dy);
    zCurr = Math.max(0, zCurr + dz);
  }

  return { t, signal, y, z };
}

function simulateToggle(tMax: number, dt: number): { t: number[]; u: number[]; v: number[] } {
  const alpha = 5;
  const n = 2;

  const nSteps = Math.round(tMax / dt);
  const t: number[] = [];
  const u: number[] = [];
  const v: number[] = [];

  // Start in the U-high state
  let uCurr = 3.5;
  let vCurr = 0.5;

  for (let i = 0; i <= nSteps; i++) {
    const time = i * dt;
    t.push(time);
    u.push(uCurr);
    v.push(vCurr);

    // Apply a pulse to V at t=5 to flip the switch
    const pulse = (time >= 5 && time < 6) ? 5 : 0;

    const du = (alpha / (1 + Math.pow(vCurr, n)) - uCurr) * dt;
    const dv = (alpha / (1 + Math.pow(uCurr, n)) + pulse - vCurr) * dt;

    uCurr = Math.max(0, uCurr + du);
    vCurr = Math.max(0, vCurr + dv);
  }

  return { t, u, v };
}

export default function MotifGallery() {
  const [selectedMotif, setSelectedMotif] = useState<MotifId>('negative-autoregulation');

  const selected = MOTIFS.find(m => m.id === selectedMotif)!;

  const chartData = useMemo(() => {
    const tMax = 20;
    const dt = 0.01;

    switch (selectedMotif) {
      case 'negative-autoregulation': {
        const { t, x, xNoFB } = simulateNAR(tMax, dt);
        return {
          traces: [
            {
              x: t, y: x, type: 'scatter', mode: 'lines',
              line: { color: '#3b82f6', width: 2.5 },
              name: 'With feedback (NAR)',
            },
            {
              x: t, y: xNoFB, type: 'scatter', mode: 'lines',
              line: { color: '#9ca3af', width: 2, dash: 'dash' },
              name: 'Without feedback',
            },
          ],
          yTitle: 'Protein X',
          yMax: Math.max(...x, ...xNoFB) * 1.15,
        };
      }

      case 'positive-feedback': {
        const { t, xLow, xHigh } = simulatePF(tMax, dt);
        return {
          traces: [
            {
              x: t, y: xHigh, type: 'scatter', mode: 'lines',
              line: { color: '#22c55e', width: 2.5 },
              name: 'Start high',
            },
            {
              x: t, y: xLow, type: 'scatter', mode: 'lines',
              line: { color: '#ef4444', width: 2.5 },
              name: 'Start low',
            },
          ],
          yTitle: 'Protein X',
          yMax: Math.max(...xHigh) * 1.15,
        };
      }

      case 'feed-forward-loop': {
        const r = simulateFFL(tMax, dt);
        return {
          traces: [
            {
              x: r.t, y: r.signal, type: 'scatter', mode: 'lines',
              line: { color: '#9ca3af', width: 1.5, dash: 'dash' },
              name: 'Signal X (scaled)',
            },
            {
              x: r.t, y: r.y, type: 'scatter', mode: 'lines',
              line: { color: '#f97316', width: 2 },
              name: 'Intermediate Y',
            },
            {
              x: r.t, y: r.z, type: 'scatter', mode: 'lines',
              line: { color: '#3b82f6', width: 2.5 },
              name: 'Output Z',
            },
          ],
          yTitle: 'Concentration',
          yMax: Math.max(...r.signal, ...r.y, ...r.z) * 1.15,
        };
      }

      case 'toggle-switch': {
        const { t, u, v } = simulateToggle(tMax, dt);
        return {
          traces: [
            {
              x: t, y: u, type: 'scatter', mode: 'lines',
              line: { color: '#a855f7', width: 2.5 },
              name: 'Gene U',
            },
            {
              x: t, y: v, type: 'scatter', mode: 'lines',
              line: { color: '#ec4899', width: 2.5 },
              name: 'Gene V',
            },
          ],
          yTitle: 'Protein level',
          yMax: Math.max(...u, ...v) * 1.15,
        };
      }
    }
  }, [selectedMotif]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Network Motif Gallery
      </h3>

      {/* Motif selector buttons */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {MOTIFS.map(motif => (
          <button
            key={motif.id}
            onClick={() => setSelectedMotif(motif.id)}
            className="text-left p-3 rounded-lg border-2 transition-colors"
            style={{
              borderColor: selectedMotif === motif.id ? motif.color : 'var(--border-strong)',
              backgroundColor: selectedMotif === motif.id ? 'var(--surface-2)' : 'transparent',
            }}
          >
            <div className="text-sm font-semibold" style={{ color: motif.color }}>
              {motif.shortName}
            </div>
            <div className="text-xs text-[var(--text-muted)] mt-0.5">
              {motif.personality}
            </div>
          </button>
        ))}
      </div>

      {/* Selected motif info */}
      <div className="mb-4 p-3 rounded-lg bg-[var(--surface-2)]">
        <div className="font-semibold text-[var(--text-strong)] mb-1">{selected.title}</div>
        <p className="text-sm text-[var(--text-muted)]">{selected.description}</p>
      </div>

      {/* Chart */}
      <CanvasChart
        data={chartData.traces as any}
        layout={{
          height: 380,
          margin: { t: 20, b: 55, l: 55, r: 20 },
          xaxis: {
            title: { text: 'Time' },
            range: [0, 20],
          },
          yaxis: {
            title: { text: chartData.yTitle },
            range: [0, chartData.yMax],
          },
          showlegend: true,
        }}
        style={{ width: '100%' }}
      />

      {/* Motif-specific observations */}
      <div className="mt-4 border-l-4 pl-4 text-sm text-[var(--text-muted)]"
        style={{ borderColor: selected.color }}>
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        {selectedMotif === 'negative-autoregulation' && (
          <p>
            The blue NAR curve overshoots briefly then settles quickly to its steady state.
            The dashed grey curve (simple production with no feedback, matched to the same steady state)
            approaches more slowly. This is the speed advantage of negative autoregulation.
          </p>
        )}
        {selectedMotif === 'positive-feedback' && (
          <p>
            Two trajectories starting from different initial conditions converge to different steady
            states &mdash; the low state and the high state. This is bistability: the system remembers
            where it started. A cell that starts high stays high; a cell that starts low stays low.
          </p>
        )}
        {selectedMotif === 'feed-forward-loop' && (
          <p>
            The signal X turns on sharply at t = 2. The intermediate Y rises gradually.
            The output Z (which requires both X AND Y) turns on with a delay &mdash; Z waits for Y to
            accumulate before it responds. When X turns off, Z drops immediately because the AND
            condition breaks. The FFL is a noise filter: brief pulses of X do not make it through.
          </p>
        )}
        {selectedMotif === 'toggle-switch' && (
          <p>
            Gene U starts high and gene V starts low. At t = 5, a brief pulse pushes V up.
            The mutual repression does the rest: V represses U, which releases V even more.
            The system flips from the U-high state to the V-high state and stays there permanently.
            This is biological memory &mdash; the pulse is gone, but the switch remembers.
          </p>
        )}
      </div>
    </div>
  );
}
