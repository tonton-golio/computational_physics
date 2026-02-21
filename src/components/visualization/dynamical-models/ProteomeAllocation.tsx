'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

/**
 * Proteome Allocation and Growth Rate
 *
 * Models the trade-off between ribosome fraction (phi_R) and metabolic
 * enzyme fraction (phi_P) in a growing cell. The housekeeping fraction
 * phi_Q is fixed overhead.
 *
 * Constraint: phi_R + phi_P + phi_Q = 1
 * Growth rate: lambda = min(k * phi_R, nu * phi_P)
 *   — limited by the slower of translation capacity or nutrient influx.
 */

function describePieSlice(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number,
): string {
  const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
  const x1 = cx + r * Math.cos(startAngle);
  const y1 = cy + r * Math.sin(startAngle);
  const x2 = cx + r * Math.cos(endAngle);
  const y2 = cy + r * Math.sin(endAngle);
  return [
    `M ${cx} ${cy}`,
    `L ${x1} ${y1}`,
    `A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`,
    'Z',
  ].join(' ');
}

export default function ProteomeAllocation() {
  const [phiQ, setPhiQ] = useState(0.4);
  const [phiR, setPhiR] = useState(0.3);
  const [nu, setNu] = useState(2.0);
  const [k, setK] = useState(2.0);

  // Clamp phiR to valid range when phiQ changes
  const phiRMax = 1 - phiQ;
  const phiRClamped = Math.min(phiR, phiRMax - 0.01);
  const phiP = Math.max(0, 1 - phiQ - phiRClamped);

  const lambda = Math.min(k * phiRClamped, nu * phiP);

  // Optimal allocation (analytical)
  const phiROpt = nu * (1 - phiQ) / (k + nu);
  const lambdaOpt = k * nu * (1 - phiQ) / (k + nu);

  // Growth rate curve: lambda vs phi_R
  const { xVals, yVals, yRibosome, yNutrient } = useMemo(() => {
    const n = 300;
    const xVals: number[] = [];
    const yVals: number[] = [];
    const yRibosome: number[] = [];
    const yNutrient: number[] = [];
    const maxPhiR = 1 - phiQ;

    for (let i = 0; i <= n; i++) {
      const pr = (i / n) * maxPhiR;
      const pp = maxPhiR - pr;
      const ribCap = k * pr;
      const nutFlux = nu * pp;
      xVals.push(pr);
      yRibosome.push(ribCap);
      yNutrient.push(nutFlux);
      yVals.push(Math.min(ribCap, nutFlux));
    }

    return { xVals, yVals, yRibosome, yNutrient };
  }, [phiQ, nu, k]);

  // SVG pie chart data
  const pieData = useMemo(() => {
    const fractions = [
      { label: '\u03C6_R', value: phiRClamped, color: '#3b82f6' },
      { label: '\u03C6_P', value: phiP, color: '#22c55e' },
      { label: '\u03C6_Q', value: phiQ, color: '#9ca3af' },
    ];
    const cx = 100;
    const cy = 100;
    const r = 85;
    let angle = -Math.PI / 2;
    const slices: Array<{
      path: string;
      color: string;
      label: string;
      value: number;
      labelX: number;
      labelY: number;
    }> = [];

    for (const f of fractions) {
      if (f.value <= 0) continue;
      const sliceAngle = f.value * 2 * Math.PI;
      const endAngle = angle + sliceAngle;
      const path = describePieSlice(cx, cy, r, angle, endAngle);
      const midAngle = angle + sliceAngle / 2;
      const labelR = r * 0.6;
      slices.push({
        path,
        color: f.color,
        label: f.label,
        value: f.value,
        labelX: cx + labelR * Math.cos(midAngle),
        labelY: cy + labelR * Math.sin(midAngle),
      });
      angle = endAngle;
    }

    return slices;
  }, [phiRClamped, phiP, phiQ]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Proteome Allocation and Growth Rate</h3>

      {/* Sliders */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Ribosome fraction \u03C6_R: {phiRClamped.toFixed(2)}
          </label>
          <Slider
            value={[phiRClamped]}
            onValueChange={([v]) => setPhiR(v)}
            min={0.05}
            max={phiRMax - 0.01}
            step={0.01}
          />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Housekeeping fraction \u03C6_Q: {phiQ.toFixed(2)}
          </label>
          <Slider
            value={[phiQ]}
            onValueChange={([v]) => {
              setPhiQ(v);
              // Clamp phiR if needed
              if (phiR > 1 - v - 0.01) setPhiR(Math.max(0.05, 1 - v - 0.01));
            }}
            min={0.2}
            max={0.6}
            step={0.05}
          />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Nutrient quality \u03BD: {nu.toFixed(1)}
          </label>
          <Slider
            value={[nu]}
            onValueChange={([v]) => setNu(v)}
            min={0.5}
            max={5}
            step={0.1}
          />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">
            Translation rate k: {k.toFixed(1)}
          </label>
          <Slider
            value={[k]}
            onValueChange={([v]) => setK(v)}
            min={0.5}
            max={5}
            step={0.1}
          />
        </div>
      </div>

      {/* Constraint display */}
      <div className="mb-4 flex flex-wrap gap-4 text-sm font-mono">
        <span className="text-[var(--text-muted)]">
          \u03C6_R = <span className="text-[#3b82f6] font-semibold">{phiRClamped.toFixed(2)}</span>
        </span>
        <span className="text-[var(--text-muted)]">
          \u03C6_P = 1 &minus; \u03C6_Q &minus; \u03C6_R = <span className="text-[#22c55e] font-semibold">{phiP.toFixed(2)}</span>
        </span>
        <span className="text-[var(--text-muted)]">
          \u03C6_Q = <span className="font-semibold" style={{ color: '#9ca3af' }}>{phiQ.toFixed(2)}</span>
        </span>
      </div>

      {/* Two visualizations side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Left: SVG pie chart */}
        <div className="flex flex-col items-center">
          <p className="text-sm text-[var(--text-muted)] mb-2">Proteome Fractions</p>
          <svg viewBox="0 0 200 200" className="w-full max-w-[280px]">
            {pieData.map((slice, i) => (
              <g key={i}>
                <path
                  d={slice.path}
                  fill={slice.color}
                  stroke="var(--surface-1)"
                  strokeWidth="1.5"
                />
                {slice.value > 0.05 && (
                  <text
                    x={slice.labelX}
                    y={slice.labelY}
                    textAnchor="middle"
                    dominantBaseline="central"
                    className="text-[10px] font-semibold"
                    fill="white"
                  >
                    {slice.label} ({(slice.value * 100).toFixed(0)}%)
                  </text>
                )}
              </g>
            ))}
          </svg>
          {/* Legend */}
          <div className="flex gap-4 mt-2 text-xs text-[var(--text-muted)]">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: '#3b82f6' }} />
              Ribosomes
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: '#22c55e' }} />
              Metabolic
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: '#9ca3af' }} />
              Housekeeping
            </span>
          </div>
        </div>

        {/* Right: Growth rate curve */}
        <div>
          <CanvasChart
            data={[
              {
                x: xVals, y: yRibosome, type: 'scatter', mode: 'lines',
                line: { color: '#3b82f6', width: 1.5, dash: 'dash' },
                name: 'k\u00B7\u03C6_R (ribosome)',
              },
              {
                x: xVals, y: yNutrient, type: 'scatter', mode: 'lines',
                line: { color: '#22c55e', width: 1.5, dash: 'dash' },
                name: '\u03BD\u00B7\u03C6_P (nutrient)',
              },
              {
                x: xVals, y: yVals, type: 'scatter', mode: 'lines',
                line: { color: '#f59e0b', width: 2.5 },
                name: '\u03BB = min(k\u03C6_R, \u03BD\u03C6_P)',
              },
              // Current operating point
              {
                x: [phiRClamped], y: [lambda], type: 'scatter', mode: 'markers',
                marker: { color: '#ef4444', size: 10 },
                name: 'Current',
              },
              // Optimal operating point
              {
                x: [phiROpt], y: [lambdaOpt], type: 'scatter', mode: 'markers',
                marker: { color: '#ffffff', size: 8, line: { width: 2, color: '#f59e0b' } },
                name: 'Optimal',
              },
            ] as any}
            layout={{
              margin: { t: 40, b: 55, l: 55, r: 20 },
              title: { text: '\u03BB vs \u03C6_R', font: { size: 14 } },
              xaxis: {
                title: { text: '\u03C6_R (ribosome fraction)' },
                range: [0, 1 - phiQ],
              },
              yaxis: {
                title: { text: 'Growth rate \u03BB' },
                range: [0, Math.max(k * (1 - phiQ), nu * (1 - phiQ)) * 1.05],
              },
              showlegend: true,
            }}
            style={{ height: 340, width: '100%' }}
          />
        </div>
      </div>

      {/* Growth rate indicator */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4 text-sm">
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">Growth rate \u03BB</div>
          <div className="text-[var(--text-strong)] font-mono text-lg">{lambda.toFixed(3)}</div>
        </div>
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">Optimal \u03BB*</div>
          <div className="text-[var(--text-strong)] font-mono text-lg">{lambdaOpt.toFixed(3)}</div>
        </div>
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">Limiting factor</div>
          <div className="text-[var(--text-strong)] font-mono text-lg">
            {k * phiRClamped < nu * phiP ? 'Ribosomes' : k * phiRClamped > nu * phiP ? 'Nutrients' : 'Balanced'}
          </div>
        </div>
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">Optimal \u03C6_R*</div>
          <div className="text-[var(--text-strong)] font-mono text-lg">{phiROpt.toFixed(3)}</div>
        </div>
      </div>

      <div className="mt-3 text-sm text-[var(--text-muted)]">
        <p>
          The cell&apos;s growth rate depends on balancing ribosomes (which make protein)
          against metabolic enzymes (which supply nutrients). More ribosomes means faster
          translation, but the cell also needs enough metabolic enzymes to keep the
          ribosomes fed. The growth rate is limited by whichever capacity is smaller:
          ribosome translation (<strong className="text-[var(--text-muted)]">k &middot; \u03C6_R</strong>)
          or nutrient influx (<strong className="text-[var(--text-muted)]">\u03BD &middot; \u03C6_P</strong>).
          The optimum occurs where both are equal, at \u03C6_R* = \u03BD(1&minus;\u03C6_Q)/(k+\u03BD).
        </p>
      </div>
    </div>
  );
}
