"use client";

import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { SimulationPanel, SimulationConfig, SimulationAux, SimulationLabel } from '@/components/ui/simulation-panel';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Creeping (Stokes) flow around a sphere. Shows streamlines at very low Re,
 * demonstrating the fore-aft symmetry unique to Stokes flow, and plots
 * Stokes drag D = 6*pi*eta*a*U vs velocity (linear scaling).
 */

export default function StokesFlowDemo({}: SimulationComponentProps) {
  const [viscosity, setViscosity] = useState(1.0); // Pa.s
  const [radius, setRadius] = useState(0.01); // m
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Stokes stream function: psi = U * sin^2(theta) * (r/2 - 3a/(4) + a^3/(4r^2))
  // In Cartesian with flow along x, sphere at origin:
  // v_r = U*cos(theta)*(1 - 3a/(2r) + a^3/(2r^3))
  // v_theta = -U*sin(theta)*(1 - 3a/(4r) - a^3/(4r^3))

  const drawStreamlines = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = Math.min(350, w * 0.55);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const cx = w * 0.45;
    const cy = h / 2;
    const scale = Math.min(w, h) * 0.08; // pixels per unit radius
    const a = 1; // normalised sphere radius
    const U = 1;

    // Draw sphere
    ctx.fillStyle = 'rgba(245,158,11,0.3)';
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, a * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Streamlines by integrating velocity field
    const nStreams = 12;
    ctx.strokeStyle = 'rgba(59,130,246,0.6)';
    ctx.lineWidth = 1.2;

    for (let si = 0; si < nStreams; si++) {
      const y0 = (si - nStreams / 2 + 0.5) * 0.5;
      let px = -5;
      let py = y0;

      ctx.beginPath();
      let started = false;

      for (let step = 0; step < 2000; step++) {
        const r = Math.sqrt(px * px + py * py);
        const dt = 0.03;

        if (r < a * 1.01) {
          // Inside sphere, no flow
          px += dt;
          continue;
        }

        const cosT = px / r;
        const sinT = py / r;
        const ra = a / r;

        const vr = U * cosT * (1 - 1.5 * ra + 0.5 * ra * ra * ra);
        const vt = -U * sinT * (1 - 0.75 * ra - 0.25 * ra * ra * ra);

        const vx = vr * cosT - vt * sinT;
        const vy = vr * sinT + vt * cosT;

        px += vx * dt;
        py += vy * dt;

        const sx = cx + px * scale;
        const sy = cy - py * scale;

        if (sx < -10 || sx > w + 10 || sy < -10 || sy > h + 10) break;

        if (!started) { ctx.moveTo(sx, sy); started = true; }
        else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
    }

    // Flow direction arrow
    ctx.fillStyle = 'rgba(148,163,184,0.6)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('U \u2192', cx + 4 * scale, 16);

    // Symmetry label
    ctx.fillStyle = 'rgba(148,163,184,0.5)';
    ctx.font = '11px sans-serif';
    ctx.fillText('Fore-aft symmetric (Stokes flow)', cx, h - 8);
  }, []);

  useEffect(() => {
    drawStreamlines();
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(drawStreamlines);
    ro.observe(container);
    return () => ro.disconnect();
  }, [drawStreamlines]);

  // Drag vs velocity chart
  const dragData = useMemo(() => {
    const vArr: number[] = [];
    const dStokes: number[] = [];
    const dQuad: number[] = [];
    const eta = viscosity;
    const a = radius;

    for (let i = 0; i <= 100; i++) {
      const v = i * 0.1;
      vArr.push(v);
      dStokes.push(6 * Math.PI * eta * a * v);
      dQuad.push(0.5 * 0.47 * 1000 * Math.PI * a * a * v * v); // turbulent Cd~0.47
    }

    return [
      {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: vArr,
        y: dStokes,
        name: 'Stokes drag (D \u221d v)',
        line: { color: '#3b82f6', width: 2 },
      },
      {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: vArr,
        y: dQuad,
        name: 'Turbulent drag (D \u221d v\u00b2)',
        line: { color: '#ef4444', width: 2, dash: 'dash' as const },
      },
    ];
  }, [viscosity, radius]);

  return (
    <SimulationPanel title="Stokes (Creeping) Flow Around a Sphere" caption="At Re \u00ab 1, viscosity dominates and the flow is fore-aft symmetric. Drag scales linearly with velocity: D = 6\u03c0\u03b7aU (Stokes' law), unlike turbulent drag which scales as v\u00b2.">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Viscosity &eta;: {viscosity.toFixed(2)} Pa&middot;s
          </SimulationLabel>
          <Slider min={0.01} max={10} step={0.01} value={[viscosity]}
            onValueChange={([v]) => setViscosity(v)} className="w-full" />
        </div>
        <div>
          <SimulationLabel>
            Sphere radius a: {(radius * 1000).toFixed(1)} mm
          </SimulationLabel>
          <Slider min={0.001} max={0.05} step={0.001} value={[radius]}
            onValueChange={([v]) => setRadius(v)} className="w-full" />
        </div>
      </SimulationConfig>
      <SimulationMain scaleMode="contain" className="w-full">
        <div ref={containerRef} style={{ width: '100%' }}>
          <canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} />
        </div>
      </SimulationMain>
      <SimulationAux>
        <CanvasChart
          data={dragData}
          layout={{
            xaxis: { title: { text: 'Velocity (m/s)' } },
            yaxis: { title: { text: 'Drag force (N)' } },
          }}
          style={{ width: '100%', height: 300 }}
        />
      </SimulationAux>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Blue: Stokes drag (linear in v). Red dashed: turbulent drag (quadratic in v).
        At low velocities Stokes drag dominates; at higher speeds the turbulent
        regime takes over.
      </p>
    </SimulationPanel>
  );
}
