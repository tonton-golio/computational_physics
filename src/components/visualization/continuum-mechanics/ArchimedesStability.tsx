'use client';

import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';

/**
 * Floating body stability visualisation. Shows a rectangular cross-section
 * floating in water. Adjusting the density ratio changes the draft (submersion),
 * and a tilt slider reveals the metacentric restoring torque. The buoyancy
 * centre, centre of gravity, and metacentre are drawn to illustrate stability.
 */

export default function ArchimedesStability() {
  const [densityRatio, setDensityRatio] = useState(0.6); // rho_obj / rho_water
  const [tiltDeg, setTiltDeg] = useState(0);
  const [bodyWidth, setBodyWidth] = useState(2.0);
  const bodyHeight = 2.0;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = Math.min(440, w * 0.65);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const scale = Math.min(w, h) * 0.14;
    const cx = w / 2;
    const waterY = h * 0.55; // waterline y-pixel

    // Draft: fraction of body submerged = density ratio
    const draft = densityRatio * bodyHeight;
    const freeboard = bodyHeight - draft;
    const bw = bodyWidth;
    const bh = bodyHeight;
    const theta = (tiltDeg * Math.PI) / 180;

    // Body corners (centered at body geometric center, which is at waterline offset)
    // The body's geometric center is at waterY - (freeboard - bh/2)*scale
    const bodyCY = waterY - (freeboard - bh / 2) * scale;
    const bodyCX = cx;

    // Centre of gravity (geometric center of uniform body)
    const gX = bodyCX;
    const gY = bodyCY;

    // Centre of buoyancy (geometric center of submerged part)
    // For untilted: at depth draft/2 below waterline
    const bCenterY = waterY + (draft / 2) * scale;

    // Metacentric height: BM = I / V_sub, I = bw^3 * L / 12, V = bw * draft * L
    // BM = bw^2 / (12 * draft)
    const BM = (bw * bw) / (12 * draft);
    // KB = draft / 2 (from keel to B)
    // KG = bh / 2 (from keel to G, keel is bottom of body)
    const KB = draft / 2;
    const KG = bh / 2;
    const GM = KB + BM - KG;

    // Water background
    ctx.fillStyle = 'rgba(59,130,246,0.08)';
    ctx.fillRect(0, waterY, w, h - waterY);

    // Waterline
    ctx.strokeStyle = 'rgba(59,130,246,0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, waterY);
    ctx.lineTo(w, waterY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw body (rotated)
    ctx.save();
    ctx.translate(bodyCX, bodyCY);
    ctx.rotate(-theta);

    // Body rectangle
    const rx = -bw / 2 * scale;
    const ry = -bh / 2 * scale;
    const rw = bw * scale;
    const rh = bh * scale;

    ctx.fillStyle = 'rgba(245,158,11,0.25)';
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx, ry, rw, rh);

    ctx.restore();

    // Draw G (centre of gravity)
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(gX, gY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(239,68,68,0.9)';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('G', gX + 8, gY + 4);

    // Draw B (centre of buoyancy)
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(bodyCX, bCenterY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(59,130,246,0.9)';
    ctx.fillText('B', bodyCX + 8, bCenterY + 4);

    // Draw M (metacentre) — BM above B
    const mY = bCenterY - BM * scale;
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(bodyCX, mY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(16,185,129,0.9)';
    ctx.fillText('M', bodyCX + 8, mY + 4);

    // Dashed line from B to M
    ctx.strokeStyle = 'rgba(148,163,184,0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(bodyCX, bCenterY);
    ctx.lineTo(bodyCX, mY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Info
    ctx.fillStyle = 'rgba(203,213,225,0.8)';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    const info = [
      `Draft = ${draft.toFixed(2)} m`,
      `GM = ${GM.toFixed(3)} m  (${GM > 0 ? 'stable' : 'unstable'})`,
      `Tilt = ${tiltDeg.toFixed(1)}\u00b0`,
    ];
    info.forEach((txt, i) => ctx.fillText(txt, 12, 20 + i * 18));

    // Stability indicator
    if (GM < 0) {
      ctx.fillStyle = 'rgba(239,68,68,0.8)';
      ctx.font = 'bold 14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('UNSTABLE (G above M)', w / 2, h - 12);
    }
  }, [densityRatio, tiltDeg, bodyWidth, bodyHeight]);

  const deps = useMemo(() => ({ densityRatio, tiltDeg, bodyWidth }), [densityRatio, tiltDeg, bodyWidth]);

  useEffect(() => {
    draw();
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(draw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [draw, deps]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-2 text-[var(--text-strong)]">
        Archimedes&apos; Principle and Floating Stability
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        A rectangular body floats according to its density ratio &rho;<sub>obj</sub>/&rho;<sub>water</sub>.
        The metacentre M must lie above the centre of gravity G for the body to be stable.
        Tall, narrow bodies can become top-heavy and capsize.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">
            Density ratio: {densityRatio.toFixed(2)}
          </label>
          <Slider min={0.1} max={0.95} step={0.01} value={[densityRatio]}
            onValueChange={([v]) => setDensityRatio(v)} className="w-full" />
        </div>
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">
            Body width: {bodyWidth.toFixed(1)} m
          </label>
          <Slider min={0.5} max={4} step={0.1} value={[bodyWidth]}
            onValueChange={([v]) => setBodyWidth(v)} className="w-full" />
        </div>
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">
            Tilt angle: {tiltDeg.toFixed(1)}&deg;
          </label>
          <Slider min={-30} max={30} step={0.5} value={[tiltDeg]}
            onValueChange={([v]) => setTiltDeg(v)} className="w-full" />
        </div>
      </div>
      <div ref={containerRef} style={{ width: '100%' }}>
        <canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} />
      </div>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        G = centre of gravity (red), B = centre of buoyancy (blue), M = metacentre (green).
        Stability requires GM &gt; 0, i.e. M above G. Try making the body narrower until it becomes unstable.
      </p>
    </div>
  );
}
