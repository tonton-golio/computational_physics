'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';

/**
 * Plots the full gravity-capillary wave dispersion relation:
 *   omega^2 = (g*k + (sigma/rho)*k^3) * tanh(k*D)
 *
 * Shows phase velocity c_p = omega/k and group velocity c_g = domega/dk.
 * A slider for water depth D transitions between deep-water and shallow-water limits.
 */

const g = 9.81;
const sigma = 0.073; // N/m surface tension (water-air)
const rho = 1000;

export default function DispersionRelation() {
  const [depth, setDepth] = useState(100); // metres
  const [showCapillary, setShowCapillary] = useState(true);

  const { omegaTraces, velocityTraces } = useMemo(() => {
    const N = 300;
    const kMin = 0.01;
    const kMax = 500; // rad/m — extend into capillary range

    const kArr: number[] = [];
    const omegaGrav: number[] = [];
    const omegaFull: number[] = [];
    const cpGrav: number[] = [];
    const cpFull: number[] = [];
    const cgFull: number[] = [];

    for (let i = 0; i <= N; i++) {
      const k = kMin * Math.pow(kMax / kMin, i / N); // log spacing
      kArr.push(k);

      const tanhKD = Math.tanh(k * depth);

      // Pure gravity
      const w2g = g * k * tanhKD;
      const wg = Math.sqrt(w2g);
      omegaGrav.push(wg);
      cpGrav.push(wg / k);

      // Full dispersion (gravity + capillary)
      const w2f = (g * k + (sigma / rho) * k * k * k) * tanhKD;
      const wf = Math.sqrt(w2f);
      omegaFull.push(wf);
      cpFull.push(wf / k);

      // Group velocity via finite difference
      const dk = k * 0.001;
      const kp = k + dk;
      const w2fp = (g * kp + (sigma / rho) * kp ** 3) * Math.tanh(kp * depth);
      const wfp = Math.sqrt(w2fp);
      cgFull.push((wfp - wf) / dk);
    }

    const omegaTraces: ChartTrace[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x: kArr,
        y: omegaGrav,
        name: 'Gravity only',
        line: { color: '#3b82f6', width: 2 },
      },
    ];
    if (showCapillary) {
      omegaTraces.push({
        type: 'scatter',
        mode: 'lines',
        x: kArr,
        y: omegaFull,
        name: 'Gravity + capillary',
        line: { color: '#ef4444', width: 2 },
      });
    }

    const velocityTraces: ChartTrace[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x: kArr,
        y: cpGrav,
        name: 'Phase vel. (gravity)',
        line: { color: '#3b82f6', width: 2 },
      },
    ];
    if (showCapillary) {
      velocityTraces.push({
        type: 'scatter',
        mode: 'lines',
        x: kArr,
        y: cpFull,
        name: 'Phase vel. (full)',
        line: { color: '#ef4444', width: 2 },
      });
      velocityTraces.push({
        type: 'scatter',
        mode: 'lines',
        x: kArr,
        y: cgFull,
        name: 'Group vel. (full)',
        line: { color: '#10b981', width: 2, dash: 'dash' },
      });
    }

    return { omegaTraces, velocityTraces };
  }, [depth, showCapillary]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-2 text-[var(--text-strong)]">
        Gravity-Capillary Wave Dispersion Relation
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        The dispersion relation &omega;&sup2; = (gk + (&sigma;/&rho;)k&sup3;)tanh(kD)
        determines how wave frequency depends on wavenumber. At low k, gravity
        dominates; at high k, surface tension takes over. In shallow water
        (small D), waves become non-dispersive.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">
            Water depth D: {depth.toFixed(0)} m
          </label>
          <Slider min={0.1} max={500} step={0.5} value={[depth]}
            onValueChange={([v]) => setDepth(v)} className="w-full" />
        </div>
        <div className="flex items-end">
          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer select-none">
            <input type="checkbox" checked={showCapillary}
              onChange={(e) => setShowCapillary(e.target.checked)} className="rounded" />
            Include capillary waves
          </label>
        </div>
      </div>
      <div className="space-y-4">
        <CanvasChart
          data={omegaTraces}
          layout={{
            xaxis: { title: { text: 'Wavenumber k (rad/m)' }, type: 'log' },
            yaxis: { title: { text: '\u03c9 (rad/s)' }, type: 'log' },
          }}
          style={{ width: '100%', height: 300 }}
        />
        <CanvasChart
          data={velocityTraces}
          layout={{
            xaxis: { title: { text: 'Wavenumber k (rad/m)' }, type: 'log' },
            yaxis: { title: { text: 'Velocity (m/s)' }, type: 'log' },
          }}
          style={{ width: '100%', height: 300 }}
        />
      </div>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Top: dispersion relation. Bottom: phase and group velocities. The
        minimum phase velocity near k ~ 370 rad/m marks the crossover between
        gravity waves and capillary waves. Reduce depth to see the shallow-water
        limit where c &rarr; &radic;(gD).
      </p>
    </div>
  );
}
