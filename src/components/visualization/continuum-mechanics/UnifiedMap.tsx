"use client";

import { useRef, useEffect, useCallback } from 'react';
import { SimulationPanel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Overview "map" of continuum mechanics showing connections between key concepts.
 * Rendered as a node-link diagram on canvas. Nodes are positioned by hand to
 * reflect the pedagogical flow of the course.
 */

interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
  color: string;
  chapter?: string;
}

interface Edge {
  from: string;
  to: string;
  label?: string;
}

const NODES: Node[] = [
  { id: 'cont', label: 'Continuum\nApproximation', x: 0.12, y: 0.15, color: '#3b82f6', chapter: 'Ch 1' },
  { id: 'tens', label: 'Stress &\nStrain Tensors', x: 0.35, y: 0.12, color: '#8b5cf6', chapter: 'Ch 2-3' },
  { id: 'hook', label: "Hooke's Law\n(Constitutive)", x: 0.55, y: 0.08, color: '#10b981', chapter: 'Ch 3' },
  { id: 'cauc', label: "Cauchy's\nEquation", x: 0.5, y: 0.35, color: '#ef4444', chapter: 'Ch 4' },
  { id: 'euler', label: "Euler's\nEquation", x: 0.25, y: 0.52, color: '#f59e0b', chapter: 'Ch 5' },
  { id: 'bern', label: "Bernoulli's\nTheorem", x: 0.08, y: 0.65, color: '#f59e0b', chapter: 'Ch 5' },
  { id: 'ns', label: 'Navier-\nStokes', x: 0.50, y: 0.55, color: '#ec4899', chapter: 'Ch 6' },
  { id: 'pois', label: 'Pipe &\nChannel Flow', x: 0.75, y: 0.48, color: '#ec4899', chapter: 'Ch 7' },
  { id: 'grav', label: 'Gravity\nWaves', x: 0.30, y: 0.75, color: '#06b6d4', chapter: 'Ch 8' },
  { id: 'stok', label: 'Stokes\n(Creeping) Flow', x: 0.70, y: 0.70, color: '#84cc16', chapter: 'Ch 9' },
  { id: 'weak', label: 'Weak Form\n& FEM', x: 0.88, y: 0.85, color: '#f97316', chapter: 'Ch 10-11' },
  { id: 'nav', label: 'Navier-Cauchy\n(Elastic)', x: 0.80, y: 0.20, color: '#10b981', chapter: 'Ch 3-4' },
  { id: 'glac', label: 'Glacier\nSimulation', x: 0.55, y: 0.88, color: '#f97316', chapter: 'Ch 12' },
];

const EDGES: Edge[] = [
  { from: 'cont', to: 'tens', label: 'smooth fields' },
  { from: 'tens', to: 'hook', label: 'linear' },
  { from: 'tens', to: 'cauc', label: 'forces' },
  { from: 'hook', to: 'cauc' },
  { from: 'hook', to: 'nav', label: 'solids' },
  { from: 'cauc', to: 'euler', label: 'inviscid' },
  { from: 'cauc', to: 'ns', label: 'viscous' },
  { from: 'euler', to: 'bern', label: 'steady' },
  { from: 'ns', to: 'pois', label: 'exact' },
  { from: 'ns', to: 'grav', label: 'linearise' },
  { from: 'ns', to: 'stok', label: 'Re\u226a1' },
  { from: 'stok', to: 'weak', label: 'numerics' },
  { from: 'weak', to: 'glac', label: 'apply' },
  { from: 'stok', to: 'glac', label: 'power-law' },
];

export default function UnifiedMap({}: SimulationComponentProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = Math.min(560, w * 0.7);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const pad = 50;
    const pw = w - 2 * pad;
    const ph = h - 2 * pad;

    const nodeMap = new Map<string, Node>();
    for (const n of NODES) nodeMap.set(n.id, n);

    // Draw edges
    for (const e of EDGES) {
      const from = nodeMap.get(e.from);
      const to = nodeMap.get(e.to);
      if (!from || !to) continue;

      const x0 = pad + from.x * pw;
      const y0 = pad + from.y * ph;
      const x1 = pad + to.x * pw;
      const y1 = pad + to.y * ph;

      ctx.strokeStyle = 'rgba(148,163,184,0.35)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(y1 - y0, x1 - x0);
      const arrowLen = 8;
      const headX = x1 - 20 * Math.cos(angle);
      const headY = y1 - 20 * Math.sin(angle);
      ctx.fillStyle = 'rgba(148,163,184,0.5)';
      ctx.beginPath();
      ctx.moveTo(headX, headY);
      ctx.lineTo(headX - arrowLen * Math.cos(angle - 0.4), headY - arrowLen * Math.sin(angle - 0.4));
      ctx.lineTo(headX - arrowLen * Math.cos(angle + 0.4), headY - arrowLen * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fill();

      // Edge label
      if (e.label) {
        const mx = (x0 + x1) / 2;
        const my = (y0 + y1) / 2;
        ctx.fillStyle = 'rgba(148,163,184,0.6)';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(e.label, mx, my - 8);
      }
    }

    // Draw nodes
    for (const n of NODES) {
      const x = pad + n.x * pw;
      const y = pad + n.y * ph;

      // Background circle
      ctx.fillStyle = n.color + '20';
      ctx.strokeStyle = n.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 32, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = 'rgba(226,232,240,0.95)';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const lines = n.label.split('\n');
      lines.forEach((line, i) => {
        ctx.fillText(line, x, y + (i - (lines.length - 1) / 2) * 12);
      });

      // Chapter tag
      if (n.chapter) {
        ctx.fillStyle = 'rgba(148,163,184,0.5)';
        ctx.font = '9px sans-serif';
        ctx.fillText(n.chapter, x, y + 28);
      }
    }
  }, []);

  useEffect(() => {
    draw();
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(draw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [draw]);

  return (
    <SimulationPanel title="The Map of Continuum Mechanics" caption="How the key concepts connect. Every path starts from the continuum approximation, passes through tensors and Cauchy's equation, then branches into solids, inviscid flow, viscous flow, and numerical methods.">
      <SimulationMain scaleMode="contain">
        <div ref={containerRef} style={{ width: '100%' }}>
          <canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} />
        </div>
      </SimulationMain>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Arrows show the logical dependency between topics. The constitutive law
        (Hooke&apos;s law for solids, Newtonian viscosity for fluids) is the fork
        that separates the elastic and fluid branches.
      </p>
    </SimulationPanel>
  );
}
