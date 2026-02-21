'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

/**
 * Velocity profile development in a laminar boundary layer over a flat plate.
 * Shows the Blasius self-similar profile u/U = f'(eta), where eta = y * sqrt(U/(nu*x)).
 * Multiple x-stations are plotted to show how the boundary layer thickens downstream.
 */

// Pre-computed Blasius profile f'(eta) using shooting method
function blasiusProfile(): { eta: number[]; fp: number[] } {
  // Solve f''' + 0.5 * f * f'' = 0, f(0)=0, f'(0)=0, f'(inf)=1
  // Use RK4 with shooting: f''(0) = 0.33206 (known exact value)
  const N = 500;
  const etaMax = 8;
  const dEta = etaMax / N;
  const eta: number[] = [];
  const fp: number[] = [];

  let f = 0, f1 = 0, f2 = 0.33206;
  for (let i = 0; i <= N; i++) {
    eta.push(i * dEta);
    fp.push(f1);

    // RK4 step for [f, f', f''] with f''' = -0.5*f*f''
    const k1f = f1 * dEta;
    const k1f1 = f2 * dEta;
    const k1f2 = -0.5 * f * f2 * dEta;

    const k2f = (f1 + k1f1 / 2) * dEta;
    const k2f1 = (f2 + k1f2 / 2) * dEta;
    const k2f2 = -0.5 * (f + k1f / 2) * (f2 + k1f2 / 2) * dEta;

    const k3f = (f1 + k2f1 / 2) * dEta;
    const k3f1 = (f2 + k2f2 / 2) * dEta;
    const k3f2 = -0.5 * (f + k2f / 2) * (f2 + k2f2 / 2) * dEta;

    const k4f = (f1 + k3f1) * dEta;
    const k4f1 = (f2 + k3f2) * dEta;
    const k4f2 = -0.5 * (f + k3f) * (f2 + k3f2) * dEta;

    f += (k1f + 2 * k2f + 2 * k3f + k4f) / 6;
    f1 += (k1f1 + 2 * k2f1 + 2 * k3f1 + k4f1) / 6;
    f2 += (k1f2 + 2 * k2f2 + 2 * k3f2 + k4f2) / 6;
  }

  return { eta, fp };
}

const BLASIUS = blasiusProfile();
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

export default function BoundaryLayer() {
  const [reL, setReL] = useState(1e5); // Reynolds number based on plate length
  const [uInf] = useState(10); // free-stream velocity m/s

  const data = useMemo(() => {
    // nu = U*L / Re, but we only need delta(x) ~ 5*x / sqrt(Rex)
    // Show profiles at several x stations
    const L = 1; // plate length
    const nu = uInf * L / reL;
    const stations = [0.05, 0.15, 0.3, 0.5, 0.8, 1.0];

    return stations.map((xFrac, idx) => {
      const x = xFrac * L;
      const Rex = uInf * x / nu;
      const delta = 5 * x / Math.sqrt(Rex); // 99% BL thickness

      // Convert Blasius eta back to physical y
      const yArr: number[] = [];
      const uArr: number[] = [];
      for (let i = 0; i < BLASIUS.eta.length; i++) {
        const y = BLASIUS.eta[i] * Math.sqrt(nu * x / uInf);
        if (y > delta * 1.5) break;
        yArr.push(y * 1000); // mm
        uArr.push(BLASIUS.fp[i] * uInf);
      }

      return {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: uArr,
        y: yArr,
        name: `x/${L} = ${xFrac}`,
        line: { color: COLORS[idx % COLORS.length], width: 2 },
      };
    });
  }, [reL, uInf]);

  const layout = useMemo(() => ({
    xaxis: { title: { text: 'Velocity u (m/s)' } },
    yaxis: { title: { text: 'Height y (mm)' } },
  }), []);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-2 text-[var(--text-strong)]">
        Boundary Layer Velocity Profiles (Blasius Solution)
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        Velocity profiles at different stations along a flat plate. The boundary
        layer thickens as &delta; ~ x/&radic;Re<sub>x</sub>. All profiles
        collapse to the same Blasius curve when plotted in similarity coordinates.
      </p>
      <div className="max-w-xs mb-4">
        <label className="block text-sm text-[var(--text-muted)] mb-1">
          Re<sub>L</sub>: {reL.toExponential(1)}
        </label>
        <Slider min={1e4} max={1e6} step={1e4} value={[reL]}
          onValueChange={([v]) => setReL(v)} className="w-full" />
      </div>
      <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 420 }} />
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Each curve shows the velocity profile at a different downstream position.
        Near the leading edge the boundary layer is thin; downstream it grows.
        Higher Re<sub>L</sub> means thinner boundary layers.
      </p>
    </div>
  );
}
