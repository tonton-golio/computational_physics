'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

/**
 * Chemotaxis: Perfect Adaptation
 *
 * Simplified two-variable model of bacterial chemotaxis receptor activity:
 *   A(t) = (1 + M(t)) / (1 + L(t))        — receptor activity
 *   dM/dt = kR - kB * A(t)                  — methylation dynamics
 *
 * Ligand signal L(t):
 *   L = L_base             for t < t_step
 *   L = L_base + deltaL    for t >= t_step
 *
 * At steady state dM/dt = 0  =>  A* = kR / kB, independent of L.
 * This is "perfect adaptation": the activity always returns to the
 * same setpoint regardless of the ambient ligand level.
 */

interface SimulationResult {
  time: number[];
  activity: number[];
  methylation: number[];
}

function simulate(
  deltaL: number,
  rateScale: number,
  lBase: number,
  dt: number,
  totalTime: number,
  tStep: number,
): SimulationResult {
  const kR = 1.0 * rateScale;
  const kB = 1.0 * rateScale;
  const steps = Math.round(totalTime / dt);

  // Initial condition: steady state with L = lBase
  // A* = kR/kB = 1, so (1 + M0)/(1 + lBase) = 1 => M0 = lBase
  let M = kR / kB * (1 + lBase) - 1;

  const time: number[] = [];
  const activity: number[] = [];
  const methylation: number[] = [];

  for (let i = 0; i <= steps; i++) {
    const t = i * dt;
    const L = t < tStep ? lBase : lBase + deltaL;
    const A = (1 + M) / (1 + L);

    // Downsample: record every 10th point to keep arrays manageable
    if (i % 10 === 0) {
      time.push(t);
      activity.push(A);
      methylation.push(M);
    }

    // Euler step for methylation
    const dMdt = kR - kB * A;
    M += dMdt * dt;
  }

  return { time, activity, methylation };
}

export default function ChemotaxisAdaptation() {
  const [deltaL, setDeltaL] = useState(3.0);
  const [rateScale, setRateScale] = useState(1.0);
  const [lBase, setLBase] = useState(1.0);

  const dt = 0.01;
  const totalTime = 30;
  const tStep = 5;

  const { time, activity, methylation } = useMemo(
    () => simulate(deltaL, rateScale, lBase, dt, totalTime, tStep),
    [deltaL, rateScale, lBase],
  );

  const steadyStateA = (1.0 * rateScale) / (1.0 * rateScale); // kR/kB = 1 always

  const chartData = useMemo(() => [
    // Activity trace
    {
      x: time,
      y: activity,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#3b82f6', width: 2.5 },
      name: 'Activity A(t)',
    },
    // Methylation trace
    {
      x: time,
      y: methylation,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#f97316', width: 2.5 },
      name: 'Methylation M(t)',
    },
    // Steady-state reference line A* = 1
    {
      x: [0, totalTime],
      y: [steadyStateA, steadyStateA],
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#6b7280', width: 1, dash: 'dash' },
      name: 'A* = kR/kB',
    },
    // "Perfect Adaptation" text annotation near the end of the trace
    {
      x: [totalTime * 0.75],
      y: [steadyStateA + 0.15],
      type: 'scatter' as const,
      mode: 'text' as const,
      text: ['Perfect Adaptation'],
      textposition: 'top center',
      showlegend: false,
    },
  ], [time, activity, methylation, steadyStateA]);

  // Compute y-axis range to fit both A(t) and M(t)
  const yRange = useMemo(() => {
    const allY = [...activity, ...methylation];
    const yMin = Math.min(...allY);
    const yMax = Math.max(...allY);
    const pad = (yMax - yMin) * 0.1 || 0.5;
    return [Math.max(0, yMin - pad), yMax + pad];
  }, [activity, methylation]);

  const chartLayout = useMemo(() => ({
    height: 420,
    margin: { t: 40, b: 60, l: 60, r: 20 },
    xaxis: {
      title: { text: 'Time' },
      range: [0, totalTime],
    },
    yaxis: {
      title: { text: 'Activity / Methylation level' },
      range: yRange,
    },
    showlegend: true,
    shapes: [
      // Vertical dashed line at t_step
      {
        type: 'line' as const,
        x0: tStep,
        x1: tStep,
        y0: 'min' as const,
        y1: 'max' as const,
        line: { color: '#ef4444', width: 1.5, dash: 'dash' },
      },
    ],
  }), [yRange]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Chemotaxis: Perfect Adaptation
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Ligand step size (Delta L): {deltaL.toFixed(1)}
          </label>
          <Slider
            value={[deltaL]}
            onValueChange={([v]) => setDeltaL(v)}
            min={0.5}
            max={10}
            step={0.5}
          />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Adaptation rate scale: {rateScale.toFixed(1)}
          </label>
          <Slider
            value={[rateScale]}
            onValueChange={([v]) => setRateScale(v)}
            min={0.2}
            max={3.0}
            step={0.1}
          />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Base ligand (L_base): {lBase.toFixed(1)}
          </label>
          <Slider
            value={[lBase]}
            onValueChange={([v]) => setLBase(v)}
            min={0.5}
            max={5}
            step={0.5}
          />
        </div>
      </div>

      <CanvasChart
        data={chartData as any}
        layout={chartLayout}
        style={{ width: '100%' }}
      />

      <div className="mt-3 flex items-center gap-2 text-sm font-medium text-[var(--text-strong)]">
        <span className="inline-block w-3 h-3 rounded-full bg-[#3b82f6]" />
        Steady-state activity A* = kR/kB = {steadyStateA.toFixed(2)} (independent of L)
      </div>

      <div className="mt-3 text-sm text-[var(--text-muted)]">
        <p>
          Bacterial chemotaxis achieves <strong className="text-[var(--text-muted)]">perfect adaptation</strong>:
          after a step change in ligand concentration (red dashed line at t = {tStep}),
          the receptor activity <span className="text-[#3b82f6] font-medium">A(t)</span> transiently
          drops but returns exactly to the pre-stimulus level A* = kR/kB.
          This occurs because the methylation level <span className="text-[#f97316] font-medium">M(t)</span> acts
          as an integral-feedback controller, slowly adjusting until
          dM/dt = 0 forces A back to kR/kB regardless of the ligand concentration.
        </p>
        <p className="mt-2">
          The <em>adaptation rate scale</em> controls how quickly both methylation
          and demethylation proceed (multiplying kR and kB equally), affecting the
          speed of recovery but not the final steady state. Larger ligand steps
          produce a deeper initial dip in activity and require greater methylation
          changes to compensate, but the endpoint is always the same.
        </p>
      </div>
    </div>
  );
}
