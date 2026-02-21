'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';

/**
 * Visualises P-wave (longitudinal / compressional) vs S-wave (transverse / shear)
 * particle motion in a material. Each dot represents a material point; its
 * displacement is shown in real time to illustrate the fundamental difference
 * between the two seismic wave modes.
 */

const ROWS = 8;
const COLS = 20;

export default function PSWaveAnimation() {
  const [frequency, setFrequency] = useState(1.5);
  const [amplitude, setAmplitude] = useState(0.35);
  const [isPlaying, setIsPlaying] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const timeRef = useRef(0);
  const animRef = useRef<number | null>(null);
  const lastTsRef = useRef(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = Math.min(420, w * 0.6);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const t = timeRef.current;
    const omega = 2 * Math.PI * frequency;
    const k = omega / 2; // wave number (arbitrary normalised speed)
    const pad = 30;
    const panelH = (h - 3 * pad) / 2;
    const spacingX = (w - 2 * pad) / (COLS - 1);
    const spacingY = panelH / (ROWS + 1);
    const dotR = Math.max(2.5, spacingX * 0.15);
    const A = amplitude * spacingX * 0.8;

    // Labels
    ctx.fillStyle = 'rgba(203,213,225,0.9)';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('P-wave (longitudinal)', pad, pad - 8);
    ctx.fillText('S-wave (transverse)', pad, pad + panelH + pad - 8);

    // Arrow showing propagation direction
    ctx.strokeStyle = 'rgba(148,163,184,0.5)';
    ctx.lineWidth = 1;
    for (const yOff of [pad + panelH + 6, pad + 2 * panelH + pad + 6]) {
      ctx.beginPath();
      ctx.moveTo(pad, yOff);
      ctx.lineTo(w - pad, yOff);
      ctx.stroke();
      // arrowhead
      ctx.beginPath();
      ctx.moveTo(w - pad, yOff);
      ctx.lineTo(w - pad - 8, yOff - 4);
      ctx.lineTo(w - pad - 8, yOff + 4);
      ctx.closePath();
      ctx.fillStyle = 'rgba(148,163,184,0.5)';
      ctx.fill();
    }

    // Draw P-wave particles
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const x0 = pad + c * spacingX;
        const y0 = pad + (r + 1) * spacingY;
        const phase = k * (c / (COLS - 1)) * 10 - omega * t;
        const dx = A * Math.sin(phase);
        // Rest position (faint)
        ctx.fillStyle = 'rgba(148,163,184,0.15)';
        ctx.beginPath();
        ctx.arc(x0, y0, dotR * 0.6, 0, Math.PI * 2);
        ctx.fill();
        // Displaced position
        ctx.fillStyle = '#3b82f6';
        ctx.beginPath();
        ctx.arc(x0 + dx, y0, dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw S-wave particles
    const yBase = pad + panelH + pad;
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const x0 = pad + c * spacingX;
        const y0 = yBase + (r + 1) * spacingY;
        const phase = k * (c / (COLS - 1)) * 10 - omega * t;
        const dy = A * Math.sin(phase);
        // Rest position (faint)
        ctx.fillStyle = 'rgba(148,163,184,0.15)';
        ctx.beginPath();
        ctx.arc(x0, y0, dotR * 0.6, 0, Math.PI * 2);
        ctx.fill();
        // Displaced position
        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(x0, y0 + dy, dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [frequency, amplitude]);

  const animate = useCallback((ts: number) => {
    if (lastTsRef.current === 0) lastTsRef.current = ts;
    const dt = (ts - lastTsRef.current) / 1000;
    lastTsRef.current = ts;
    timeRef.current += dt;
    draw();
    animRef.current = requestAnimationFrame(animate);
  }, [draw]);

  useEffect(() => {
    if (isPlaying) {
      lastTsRef.current = 0;
      animRef.current = requestAnimationFrame(animate);
    } else {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      animRef.current = null;
      draw();
    }
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [isPlaying, animate, draw]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(container);
    return () => ro.disconnect();
  }, [draw]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-2 text-[var(--text-strong)]">
        P-wave vs S-wave Particle Motion
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        P-waves displace particles parallel to the propagation direction
        (compression/rarefaction). S-waves displace particles perpendicular
        to propagation (shear). Faint dots show rest positions.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">
            Frequency: {frequency.toFixed(1)}
          </label>
          <Slider min={0.5} max={4} step={0.1} value={[frequency]}
            onValueChange={([v]) => setFrequency(v)} className="w-full" />
        </div>
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">
            Amplitude: {amplitude.toFixed(2)}
          </label>
          <Slider min={0.1} max={0.6} step={0.02} value={[amplitude]}
            onValueChange={([v]) => setAmplitude(v)} className="w-full" />
        </div>
      </div>
      <div className="flex gap-3 mb-4">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            isPlaying ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white'
          }`}
        >{isPlaying ? 'Pause' : 'Play'}</button>
        <button
          onClick={() => { timeRef.current = 0; if (!isPlaying) draw(); }}
          className="px-4 py-2 rounded-lg text-sm font-medium bg-[var(--surface-3)] hover:bg-[var(--border-strong)] text-[var(--text-strong)] transition-colors"
        >Reset</button>
      </div>
      <div ref={containerRef} style={{ width: '100%' }}>
        <canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} />
      </div>
    </div>
  );
}
