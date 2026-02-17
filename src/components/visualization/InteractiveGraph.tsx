'use client';

import { useEffect, useRef, useState } from 'react';
import type Plotly from 'plotly.js-dist';

interface GraphProps {
  type: string;
  params?: Record<string, number>;
  title?: string;
}

const COLORS = {
  primary: '#3b82f6',
  secondary: '#ec4899',
  tertiary: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  accent: '#8b5cf6',
};

const BASE_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: '#0a1120',
  font: { color: '#b8c3d9', family: 'var(--font-geist-mono), ui-monospace, SFMono-Regular, Menlo, monospace' },
  margin: { t: 48, r: 22, b: 48, l: 58 },
  hoverlabel: {
    bgcolor: '#111d33',
    bordercolor: '#355487',
    font: { color: '#dbeafe' },
  },
  legend: {
    bgcolor: 'rgba(11,22,39,0.75)',
    bordercolor: '#273d63',
    borderwidth: 1,
    font: { color: '#cdd7e9' },
  },
  xaxis: { 
    gridcolor: '#1a2740',
    zerolinecolor: '#243858',
    linecolor: '#33507f',
    tickfont: { color: '#9eb0cf' },
  },
  yaxis: { 
    gridcolor: '#1a2740',
    zerolinecolor: '#243858',
    linecolor: '#33507f',
    tickfont: { color: '#9eb0cf' },
  },
};

let plotlyPromise: Promise<typeof import('plotly.js-dist')['default']> | null = null;

function loadPlotly() {
  if (plotlyPromise) return plotlyPromise;
  plotlyPromise = import('plotly.js-dist').then((module) => module.default);
  return plotlyPromise;
}

function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function withTheme(layout: Partial<Plotly.Layout>): Partial<Plotly.Layout> {
  const background = cssVar("--surface-2", "#0a1120");
  const textStrong = cssVar("--text-strong", "#f4f7ff");
  const textMuted = cssVar("--text-muted", "#9eb0cf");
  const border = cssVar("--border-strong", "#2c4166");
  const surface = cssVar("--surface-1", "#0d1528");

  return {
    ...layout,
    plot_bgcolor: background,
    font: { ...(layout.font ?? {}), color: textMuted, family: 'var(--font-geist-mono), ui-monospace, SFMono-Regular, Menlo, monospace' },
    legend: {
      ...(layout.legend ?? {}),
      bgcolor: surface,
      bordercolor: border,
      borderwidth: 1,
      font: { ...((layout.legend as Plotly.Legend)?.font ?? {}), color: textMuted },
    },
    hoverlabel: {
      ...(layout.hoverlabel ?? {}),
      bgcolor: surface,
      bordercolor: border,
      font: { ...((layout.hoverlabel as Plotly.HoverLabel)?.font ?? {}), color: textStrong },
    },
    xaxis: {
      ...((layout.xaxis as Plotly.LayoutAxis) ?? {}),
      gridcolor: border,
      zerolinecolor: border,
      linecolor: border,
      tickfont: { ...(((layout.xaxis as Plotly.LayoutAxis)?.tickfont) ?? {}), color: textMuted },
      title: {
        ...(((layout.xaxis as Plotly.LayoutAxis)?.title) ?? {}),
        font: { ...((((layout.xaxis as Plotly.LayoutAxis)?.title as Plotly.DataTitle)?.font) ?? {}), color: textStrong },
      },
    },
    yaxis: {
      ...((layout.yaxis as Plotly.LayoutAxis) ?? {}),
      gridcolor: border,
      zerolinecolor: border,
      linecolor: border,
      tickfont: { ...(((layout.yaxis as Plotly.LayoutAxis)?.tickfont) ?? {}), color: textMuted },
      title: {
        ...(((layout.yaxis as Plotly.LayoutAxis)?.title) ?? {}),
        font: { ...((((layout.yaxis as Plotly.LayoutAxis)?.title as Plotly.DataTitle)?.font) ?? {}), color: textStrong },
      },
    },
    scene: layout.scene
      ? {
          ...layout.scene,
          bgcolor: background,
          xaxis: { ...(layout.scene.xaxis ?? {}), gridcolor: border, color: textMuted },
          yaxis: { ...(layout.scene.yaxis ?? {}), gridcolor: border, color: textMuted },
          zaxis: { ...(layout.scene.zaxis ?? {}), gridcolor: border, color: textMuted },
        }
      : undefined,
  };
}

// ============ BASIC PHYSICS ============

function harmonicMotion(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const A = params.amplitude || 1;
  const omega = params.frequency || 1;
  const phi = params.phase || 0;
  const T = params.duration || 4 * Math.PI;
  
  const x = Array.from({ length: 200 }, (_, i) => (i / 200) * T);
  const y = x.map(t => A * Math.cos(omega * t + phi));
  const velocity = x.map(t => -A * omega * Math.sin(omega * t + phi));
  const acceleration = x.map(t => -A * omega * omega * Math.cos(omega * t + phi));
  
  return {
    data: [
      { x, y, type: 'scatter', mode: 'lines', line: { color: COLORS.primary, width: 2 }, name: 'x(t)' },
      { x, y: velocity, type: 'scatter', mode: 'lines', line: { color: COLORS.secondary, width: 1, dash: 'dash' }, name: 'v(t)', visible: 'legendonly' },
      { x, y: acceleration, type: 'scatter', mode: 'lines', line: { color: COLORS.tertiary, width: 1, dash: 'dot' }, name: 'a(t)', visible: 'legendonly' },
    ],
    layout: { ...BASE_LAYOUT, xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Time (s)' } }, yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Displacement' } }, title: { text: 'Simple Harmonic Motion' } },
  };
}

function dampedOscillator(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const A = params.amplitude || 1;
  const omega0 = params.frequency || 2;
  const gamma = params.damping || 0.15;
  const T = params.duration || 10;
  
  const t = Array.from({ length: 300 }, (_, i) => (i / 300) * T);
  const y = t.map(ti => A * Math.exp(-gamma * ti) * Math.cos(omega0 * ti));
  const envelope = t.map(ti => A * Math.exp(-gamma * ti));
  
  return {
    data: [
      { x: t, y, type: 'scatter', mode: 'lines', line: { color: COLORS.primary, width: 2 }, name: 'x(t)' },
      { x: t, y: envelope, type: 'scatter', mode: 'lines', line: { color: COLORS.danger, width: 1, dash: 'dash' }, name: 'Envelope' },
      { x: t, y: envelope.map(e => -e), type: 'scatter', mode: 'lines', line: { color: COLORS.danger, width: 1, dash: 'dash' }, showlegend: false },
    ],
    layout: { ...BASE_LAYOUT, xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Time (s)' } }, yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Displacement' } }, title: { text: 'Damped Harmonic Motion' } },
  };
}

function wavePropagation(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const k = params.wavenumber || 1;
  const omega = params.frequency || 1;
  const L = params.length || 4 * Math.PI;
  
  const x = Array.from({ length: 200 }, (_, i) => (i / 200) * L);
  const snapshots = [0, 0.5, 1, 1.5, 2];
  
  return {
    data: snapshots.map((t, idx) => ({
      x,
      y: x.map(xi => Math.sin(k * xi - omega * t)),
      type: 'scatter',
      mode: 'lines',
      line: { color: [COLORS.primary, COLORS.secondary, COLORS.tertiary, COLORS.warning, COLORS.accent][idx], width: 2 },
      name: `t = ${t}`,
    })),
    layout: { ...BASE_LAYOUT, xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Position x' } }, yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Amplitude' } }, title: { text: 'Wave Propagation' } },
  };
}

// ============ QUANTUM OPTICS ============

function wignerGaussian(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const mu_x = params.x_mean || 0;
  const mu_p = params.p_mean || 0;
  const sigma = params.stddev || 0.5;
  const range = params.range || 4;
  
  const x = Array.from({ length: 50 }, (_, i) => mu_x - range + (i / 50) * 2 * range);
  const p = Array.from({ length: 50 }, (_, i) => mu_p - range + (i / 50) * 2 * range);
  
  const z: number[][] = [];
  for (let i = 0; i < p.length; i++) {
    z[i] = [];
    for (let j = 0; j < x.length; j++) {
      const dx = x[j] - mu_x;
      const dp = p[i] - mu_p;
      z[i][j] = (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-(dx * dx + dp * dp) / (2 * sigma * sigma));
    }
  }
  
  return {
    data: [{
      type: 'heatmap',
      z,
      x,
      y: p,
      colorscale: 'RdBu_r',
      reversescale: true,
      colorbar: { title: { text: 'W(x,p)', side: 'right' } },
    }],
    layout: { 
      ...BASE_LAYOUT, 
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Position q' } }, 
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Momentum p' } },
      title: { text: 'Wigner Function (Coherent State)' },
    },
  };
}

function wignerNumberState(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const n = Math.floor(params.n || 0);
  const range = params.range || 6;
  
  const N = 60;
  const x = Array.from({ length: N }, (_, i) => -range + (i / N) * 2 * range);
  const p = x.slice();
  
  // Simplified Wigner for number state (approximate)
  const z: number[][] = [];
  for (let i = 0; i < p.length; i++) {
    z[i] = [];
    for (let j = 0; j < x.length; j++) {
      const r2 = x[j] * x[j] + p[i] * p[i];
      // Laguerre polynomial approximation
      const L_n = n === 0 ? 1 : n === 1 ? 1 - r2 : n === 2 ? (1 - 2 * r2 + r2 * r2 / 2) : 1;
      z[i][j] = (2 / Math.PI) * Math.pow(-1, n) * L_n * Math.exp(-r2);
    }
  }
  
  return {
    data: [{
      type: 'heatmap',
      z,
      x,
      y: p,
      colorscale: 'RdBu_r',
      reversescale: true,
      colorbar: { title: { text: 'W(q,p)', side: 'right' } },
    }],
    layout: { 
      ...BASE_LAYOUT, 
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'q' } }, 
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'p' } },
      title: { text: `Wigner Function |n=${n}⟩` },
    },
  };
}

function wignerCatState(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const alpha = params.alpha || 2;
  const range = params.range || 5;
  
  const N = 60;
  const x = Array.from({ length: N }, (_, i) => -range + (i / N) * 2 * range);
  const p = x.slice();
  
  // Cat state Wigner (two coherent states superposition)
  const z: number[][] = [];
  for (let i = 0; i < p.length; i++) {
    z[i] = [];
    for (let j = 0; j < x.length; j++) {
      const r1 = Math.pow(x[j] - alpha, 2) + p[i] * p[i];
      const r2 = Math.pow(x[j] + alpha, 2) + p[i] * p[i];
      const gauss1 = Math.exp(-r1);
      const gauss2 = Math.exp(-r2);
      const interference = Math.cos(2 * alpha * p[i]) * Math.exp(-x[j] * x[j] - p[i] * p[i]);
      z[i][j] = (2 / Math.PI) * (gauss1 + gauss2 + 2 * interference) / (2 + 2 * Math.exp(-2 * alpha * alpha));
    }
  }
  
  return {
    data: [{
      type: 'heatmap',
      z,
      x,
      y: p,
      colorscale: 'RdBu_r',
      reversescale: true,
      colorbar: { title: { text: 'W(q,p)', side: 'right' } },
    }],
    layout: { 
      ...BASE_LAYOUT, 
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'q' } }, 
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'p' } },
      title: { text: 'Wigner Function (Cat State)' },
    },
  };
}

// ============ STATISTICS ============

function gaussianDistribution(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const mu = params.mean || 0;
  const sigma = params.stddev || 1;
  
  const x = Array.from({ length: 200 }, (_, i) => mu - 4 * sigma + (i / 200) * 8 * sigma);
  const y = x.map(xi => (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((xi - mu) / sigma, 2)));
  
  return {
    data: [{
      x,
      y,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      fillcolor: 'rgba(59, 130, 246, 0.2)',
      line: { color: COLORS.primary, width: 2 },
    }],
    layout: { ...BASE_LAYOUT, xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'x' } }, yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'P(x)' } }, title: { text: 'Gaussian Distribution' } },
  };
}

function binomialDistribution(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const n = Math.floor(params.n || 20);
  const p = params.probability || 0.5;
  
  // Binomial coefficient
  const factorial = (k: number): number => k <= 1 ? 1 : k * factorial(k - 1);
  const binom = (n: number, k: number) => factorial(n) / (factorial(k) * factorial(n - k));
  
  const x = Array.from({ length: n + 1 }, (_, k) => k);
  const y = x.map(k => binom(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k));
  
  return {
    data: [{
      x,
      y,
      type: 'bar',
      marker: { color: COLORS.primary },
    }],
    layout: { ...BASE_LAYOUT, xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'k' } }, yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'P(X=k)' } }, title: { text: `Binomial(n=${n}, p=${p})` } },
  };
}

function poissonDistribution(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const lambda = params.lambda || 5;
  const maxK = Math.floor(params.max_k || 20);
  
  // Factorial
  const factorial = (k: number): number => k <= 1 ? 1 : k * factorial(k - 1);
  
  const x = Array.from({ length: maxK + 1 }, (_, k) => k);
  const y = x.map(k => (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k));
  
  return {
    data: [{
      x,
      y,
      type: 'bar',
      marker: { color: COLORS.secondary },
    }],
    layout: { ...BASE_LAYOUT, xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'k' } }, yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'P(X=k)' } }, title: { text: `Poisson(λ=${lambda})` } },
  };
}

function centralLimitTheorem(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const n = Math.floor(params.n || 100);
  const samples = Math.floor(params.samples || 1000);
  
  // Simulate means of uniform distribution
  const means: number[] = [];
  for (let s = 0; s < samples; s++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += Math.random(); // Uniform [0,1]
    }
    means.push(sum / n);
  }
  
  // Create histogram
  const bins = 40;
  const min = Math.min(...means);
  const max = Math.max(...means);
  const binWidth = (max - min) / bins;
  
  const counts: number[] = new Array(bins).fill(0);
  const binCenters: number[] = [];
  
  for (let i = 0; i < bins; i++) {
    binCenters.push(min + (i + 0.5) * binWidth);
  }
  
  means.forEach(m => {
    const binIdx = Math.min(Math.floor((m - min) / binWidth), bins - 1);
    counts[binIdx]++;
  });
  
  // Normalized for comparison
  const normalized = counts.map(c => c / samples / binWidth);
  
  // Overlay Gaussian
  const mu = 0.5;
  const sigma = Math.sqrt(1 / 12 / n);
  const gaussian = binCenters.map(x => (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)));
  
  return {
    data: [
      { x: binCenters, y: normalized, type: 'bar', marker: { color: COLORS.primary, opacity: 0.7 }, name: 'Sample means' },
      { x: binCenters, y: gaussian, type: 'scatter', mode: 'lines', line: { color: COLORS.danger, width: 2 }, name: 'Gaussian fit' },
    ],
    layout: { 
      ...BASE_LAYOUT, 
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Sample mean' } }, 
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Density' } }, 
      title: { text: `Central Limit Theorem (n=${n})` },
      bargap: 0,
    },
  };
}

// ============ CONTINUUM MECHANICS ============

function beamDeflection(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const L = params.length || 10;
  const E = params.youngs_modulus || 200;
  const I = params.moment_inertia || 1;
  const F = params.force || 100;
  const q = params.distributed_load || 0;
  
  const x = Array.from({ length: 100 }, (_, i) => (i / 100) * L);
  
  // Cantilever beam deflection
  const delta = x.map(xi => {
    const pointLoad = -(F * xi * xi) / (6 * E * I) * (3 * L - xi);
    const distributed = -(q * xi * xi) / (24 * E * I) * (xi * xi - 4 * L * xi + 6 * L * L);
    return pointLoad + distributed;
  });
  
  // Scale for visualization
  const maxDelta = Math.max(...delta.map(Math.abs));
  const scaled = delta.map(d => d / (maxDelta || 1) * L * 0.1);
  
  return {
    data: [
      { x: [0, L], y: [0, 0], type: 'scatter', mode: 'lines', line: { color: COLORS.tertiary, width: 4 }, name: 'Undeformed' },
      { x, y: scaled, type: 'scatter', mode: 'lines', line: { color: COLORS.primary, width: 3 }, name: 'Deflected' },
    ],
    layout: { 
      ...BASE_LAYOUT, 
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Position (m)' }, scaleanchor: 'y' }, 
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Deflection (scaled)' } },
      title: { text: 'Beam Deflection' },
    },
  };
}

function stressStrainCurve(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const E = params.youngs_modulus || 200;
  const yieldStress = params.yield_stress || 250;
  const ultimateStress = params.ultimate_stress || 400;
  
  const strain = Array.from({ length: 200 }, (_, i) => (i / 200) * 0.02);
  
  // Simplified stress-strain curve
  const stress = strain.map(eps => {
    const elastic = E * eps * 1000; // Convert to MPa
    if (elastic < yieldStress) return elastic;
    if (elastic < ultimateStress) {
      // Strain hardening region
      return yieldStress + (elastic - yieldStress) * 0.1;
    }
    return ultimateStress;
  });
  
  return {
    data: [{
      x: strain.map(s => s * 100), // Percent strain
      y: stress,
      type: 'scatter',
      mode: 'lines',
      line: { color: COLORS.primary, width: 2 },
      fill: 'tozeroy',
      fillcolor: 'rgba(59, 130, 246, 0.1)',
    }],
    layout: { 
      ...BASE_LAYOUT, 
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'Strain (%)' } }, 
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'Stress (MPa)' } },
      title: { text: 'Stress-Strain Curve' },
      shapes: [
        { type: 'line', x0: 0, x1: yieldStress / E * 100, y0: yieldStress, y1: yieldStress, line: { color: COLORS.secondary, dash: 'dash' } },
      ],
    },
  };
}

// ============ DYNAMICAL SYSTEMS ============

function phasePortrait(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const alpha = params.alpha || -1; // Negative = stable spiral, positive = unstable
  const omega = params.omega || 1;
  const range = params.range || 3;
  
  const N = 15;
  const x0 = Array.from({ length: N }, (_, i) => -range + (i / (N - 1)) * 2 * range);
  const y0 = x0.slice();
  
  // Create vector field
  const data: Plotly.Data[] = [];
  const vectors: { x: number[]; y: number[]; u: number[]; v: number[] } = { x: [], y: [], u: [], v: [] };
  
  for (const xi of x0) {
    for (const yi of y0) {
      vectors.x.push(xi);
      vectors.y.push(yi);
      // dx/dt = alpha * x - omega * y
      // dy/dt = omega * x + alpha * y
      const u = alpha * xi - omega * yi;
      const v = omega * xi + alpha * yi;
      const norm = Math.sqrt(u * u + v * v);
      vectors.u.push(u / (norm || 1) * 0.3);
      vectors.v.push(v / (norm || 1) * 0.3);
    }
  }
  
  // Sample trajectories
  const dt = 0.05;
  const tMax = 10;
  const trajectories = [
    { x: 2, y: 0 },
    { x: -2, y: 0 },
    { x: 0, y: 2 },
    { x: 0, y: -2 },
    { x: 1.5, y: 1.5 },
  ];
  
  trajectories.forEach((start, idx) => {
    const trajX: number[] = [start.x];
    const trajY: number[] = [start.y];
    let x = start.x, y = start.y;
    
    for (let t = 0; t < tMax; t += dt) {
      const dx = alpha * x - omega * y;
      const dy = omega * x + alpha * y;
      x += dx * dt;
      y += dy * dt;
      trajX.push(x);
      trajY.push(y);
    }
    
    data.push({
      x: trajX,
      y: trajY,
      type: 'scatter',
      mode: 'lines',
      line: { color: [COLORS.primary, COLORS.secondary, COLORS.tertiary, COLORS.warning, COLORS.accent][idx], width: 2 },
      showlegend: false,
    });
  });
  
  // Add vector field as arrows using scatter (simplified)
  for (let i = 0; i < vectors.x.length; i += 3) {
    const x = vectors.x[i];
    const y = vectors.y[i];
    const u = vectors.u[i];
    const v = vectors.v[i];
    
    data.push({
      x: [x, x + u],
      y: [y, y + v],
      type: 'scatter',
      mode: 'lines',
      line: { color: '#444', width: 1 },
      showlegend: false,
    });
  }
  
  return {
    data,
    layout: { 
      ...BASE_LAYOUT, 
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'x' }, range: [-range, range] }, 
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'y' }, range: [-range, range], scaleanchor: 'x' },
      title: { text: 'Phase Portrait' },
    },
  };
}

function lorenzAttractor(params: Record<string, number>): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const sigma = params.sigma || 10;
  const rho = params.rho || 28;
  const beta = params.beta || 8 / 3;
  
  const dt = 0.01;
  const steps = 3000;
  
  let x = 1, y = 1, z = 1;
  const trajectory: { x: number[]; y: number[]; z: number[] } = { x: [], y: [], z: [] };
  
  for (let i = 0; i < steps; i++) {
    const dx = sigma * (y - x);
    const dy = x * (rho - z) - y;
    const dz = x * y - beta * z;
    
    x += dx * dt;
    y += dy * dt;
    z += dz * dt;
    
    trajectory.x.push(x);
    trajectory.y.push(y);
    trajectory.z.push(z);
  }
  
  return {
    data: [{
      type: 'scatter3d',
      mode: 'lines',
      x: trajectory.x,
      y: trajectory.y,
      z: trajectory.z,
      line: { color: trajectory.z.map((_, i) => i / steps), width: 2 },
      marker: { colorscale: 'Viridis' },
    }],
    layout: {
      paper_bgcolor: 'rgba(0,0,0,0)',
      scene: {
        xaxis: { title: { text: 'x' }, gridcolor: '#1e1e2e', color: '#9ca3af' },
        yaxis: { title: { text: 'y' }, gridcolor: '#1e1e2e', color: '#9ca3af' },
        zaxis: { title: { text: 'z' }, gridcolor: '#1e1e2e', color: '#9ca3af' },
        bgcolor: 'rgba(15,15,25,1)',
      },
      title: { text: 'Lorenz Attractor' },
      margin: { l: 0, r: 0, b: 0, t: 40 },
    },
  };
}

// ============ GRAPH REGISTRY ============

const GRAPH_GENERATORS: Record<string, (p: Record<string, number>) => { data: Plotly.Data[]; layout: Partial<Plotly.Layout> }> = {
  // Basic Physics
  'harmonic-motion': harmonicMotion,
  'damped-oscillator': dampedOscillator,
  'wave-propagation': wavePropagation,
  
  // Quantum Optics
  'wigner-gaussian': wignerGaussian,
  'wigner-number': wignerNumberState,
  'wigner-cat': wignerCatState,
  
  // Statistics
  'gaussian': gaussianDistribution,
  'binomial': binomialDistribution,
  'poisson': poissonDistribution,
  'central-limit': centralLimitTheorem,
  
  // Continuum Mechanics
  'beam-deflection': beamDeflection,
  'stress-strain': stressStrainCurve,
  
  // Dynamical Systems
  'phase-portrait': phasePortrait,
  'lorenz': lorenzAttractor,
};

// Legacy aliases
const ALIASES: Record<string, string> = {
  'harmonic': 'harmonic-motion',
  'damped': 'damped-oscillator',
  'wave': 'wave-propagation',
  'gaussian-dist': 'gaussian',
};

export function InteractiveGraph({ type, params = {}, title }: GraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [themeVersion, setThemeVersion] = useState(0);

  useEffect(() => {
    const observer = new MutationObserver(() => setThemeVersion((v) => v + 1));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);
  
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let active = true;
    void loadPlotly().then((plotly) => {
      if (!active) return;
      const resolvedType = ALIASES[type] || type;
      const generator = GRAPH_GENERATORS[resolvedType];

      if (generator) {
        const { data, layout } = generator(params);
        const mergedLayout = title ? { ...layout, title: { text: title } } : layout;
        const finalLayout = withTheme(mergedLayout);

        plotly.newPlot(container, data, finalLayout, {
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
          displaylogo: false,
        });
        return;
      }

      plotly.newPlot(container, [], {
        ...withTheme(BASE_LAYOUT),
        title: { text: `Unknown graph type: ${type}` },
      });
    });

    return () => {
      active = false;
      void loadPlotly().then((plotly) => {
        plotly.purge(container);
      });
    };
  }, [type, params, title, themeVersion]);
  
  return (
    <div 
      ref={containerRef} 
      className="h-72 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)] p-1"
    />
  );
}

// List of available graphs for content
export const GRAPH_DEFS: Record<string, GraphProps> = {
  'harmonic-motion': {
    type: 'harmonic',
    params: { amplitude: 1, frequency: 2, phase: 0 },
  },
  'damped-oscillator': {
    type: 'damped',
    params: { amplitude: 1, frequency: 2, damping: 0.15 },
  },
  'wave-propagation': {
    type: 'wave',
    params: { wavenumber: 1, frequency: 1 },
  },
  'gaussian': {
    type: 'gaussian',
    params: { mean: 0, stddev: 1 },
  },
};
