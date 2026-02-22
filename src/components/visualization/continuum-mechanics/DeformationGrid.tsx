"use client";

import { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Shows a 2D grid deformed by different strain states: pure shear, simple shear,
 * uniaxial extension, and biaxial extension. Sliders control the strain components.
 * The grid is rendered directly on a canvas for pixel-perfect control.
 */

const GRID_N = 10;

export default function DeformationGrid({}: SimulationComponentProps) {
  const [exx, setExx] = useState(0.0);
  const [eyy, setEyy] = useState(0.0);
  const [exy, setExy] = useState(0.0);
  const [preset, setPreset] = useState('custom');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const presets: Record<string, [number, number, number, string]> = {
    'uniaxial-x': [0.3, 0, 0, 'Uniaxial extension (x)'],
    'uniaxial-y': [0, 0.3, 0, 'Uniaxial extension (y)'],
    'biaxial': [0.2, 0.2, 0, 'Biaxial extension'],
    'pure-shear': [0.2, -0.2, 0, 'Pure shear'],
    'simple-shear': [0, 0, 0.3, 'Simple shear'],
    'custom': [exx, eyy, exy, 'Custom'],
  };

  const applyPreset = (key: string) => {
    setPreset(key);
    if (key !== 'custom') {
      const [a, b, c] = presets[key];
      setExx(a); setEyy(b); setExy(c);
    }
  };

  // Deformation mapping: x' = (I + E) * x
  const deform = useCallback((x: number, y: number): [number, number] => {
    return [
      x * (1 + exx) + y * exy,
      x * exy + y * (1 + eyy),
    ];
  }, [exx, eyy, exy]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = Math.min(w, 500);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const scale = Math.min(w, h) * 0.32;

    // Draw original grid (faint)
    ctx.strokeStyle = 'rgba(148,163,184,0.2)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID_N; i++) {
      const t = -0.5 + i / GRID_N;
      ctx.beginPath();
      ctx.moveTo(cx + (-0.5) * scale, cy - t * scale);
      ctx.lineTo(cx + 0.5 * scale, cy - t * scale);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx + t * scale, cy - (-0.5) * scale);
      ctx.lineTo(cx + t * scale, cy - 0.5 * scale);
      ctx.stroke();
    }

    // Draw deformed grid
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.5;
    for (let i = 0; i <= GRID_N; i++) {
      const t = -0.5 + i / GRID_N;
      // Horizontal lines
      ctx.beginPath();
      for (let j = 0; j <= 60; j++) {
        const s = -0.5 + j / 60;
        const [dx, dy] = deform(s, t);
        const px = cx + dx * scale;
        const py = cy - dy * scale;
        if (j === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      // Vertical lines
      ctx.beginPath();
      for (let j = 0; j <= 60; j++) {
        const s = -0.5 + j / 60;
        const [dx, dy] = deform(t, s);
        const px = cx + dx * scale;
        const py = cy - dy * scale;
        if (j === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Mark origin
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();

    // Axes labels
    ctx.fillStyle = 'rgba(148,163,184,0.7)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('x', cx + 0.55 * scale, cy + 16);
    ctx.fillText('y', cx - 16, cy - 0.55 * scale);
  }, [deform]);

  const deps = useMemo(() => ({ exx, eyy, exy }), [exx, eyy, exy]);

  useEffect(() => {
    draw();
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(draw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [draw, deps]);

  return (
    <SimulationPanel title="2D Deformation Grid" caption="Apply strain components to deform a regular grid. The faint grid shows the reference configuration; the blue grid shows the deformed state.">
      <SimulationSettings>
        <SimulationToggle
          options={Object.entries(presets).filter(([k]) => k !== 'custom').map(([key, [,,,label]]) => ({
            label: label as string,
            value: key,
          }))}
          value={preset}
          onChange={(v) => applyPreset(v)}
        />
      </SimulationSettings>
      <SimulationConfig>
        <div>
          <SimulationLabel>
            &epsilon;<sub>xx</sub>: {exx.toFixed(2)}
          </SimulationLabel>
          <Slider min={-0.5} max={0.5} step={0.01} value={[exx]}
            onValueChange={([v]) => { setExx(v); setPreset('custom'); }} className="w-full" />
        </div>
        <div>
          <SimulationLabel>
            &epsilon;<sub>yy</sub>: {eyy.toFixed(2)}
          </SimulationLabel>
          <Slider min={-0.5} max={0.5} step={0.01} value={[eyy]}
            onValueChange={([v]) => { setEyy(v); setPreset('custom'); }} className="w-full" />
        </div>
        <div>
          <SimulationLabel>
            &epsilon;<sub>xy</sub> (shear): {exy.toFixed(2)}
          </SimulationLabel>
          <Slider min={-0.5} max={0.5} step={0.01} value={[exy]}
            onValueChange={([v]) => { setExy(v); setPreset('custom'); }} className="w-full" />
        </div>
      </SimulationConfig>
      <SimulationMain scaleMode="contain">
        <div ref={containerRef} style={{ width: '100%' }}>
          <canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} />
        </div>
      </SimulationMain>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        The deformation gradient is F = I + &epsilon;. For small strains the
        symmetric part gives stretch and the antisymmetric part gives rotation.
      </p>
    </SimulationPanel>
  );
}
