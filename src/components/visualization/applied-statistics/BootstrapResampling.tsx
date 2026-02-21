'use client';

import React, { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import type { SimulationComponentProps } from '@/shared/types/simulation';


/**
 * Error Propagation Area Demo.
 * Given a rectangle with uncertain width W and length L (each with Gaussian errors),
 * demonstrate how the uncertainty propagates into the area A = W * L.
 */
export default function AppliedStatsSim8({ }: SimulationComponentProps) {
  const [W, setW] = useState(5);
  const [L, setL] = useState(5);
  const [sigW, setSigW] = useState(0.3);
  const [sigL, setSigL] = useState(0.3);

  const result = useMemo(() => {
    const nSim = 2000;
    // Seeded random for reproducibility
    let s = 42;
    const seededRandom = () => {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };
    const seededNormal = () => {
      const u1 = seededRandom();
      const u2 = seededRandom();
      return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    };

    // Generate random W and L samples
    const Ws: number[] = [];
    const Ls: number[] = [];
    const areas: number[] = [];
    // Edge scatter points for width uncertainty (points along right edge)
    const edgeWx: number[] = [];
    const edgeWy: number[] = [];
    // Edge scatter points for length uncertainty (points along top edge)
    const edgeLx: number[] = [];
    const edgeLy: number[] = [];

    for (let i = 0; i < nSim; i++) {
      const wi = W + sigW * seededNormal();
      const li = L + sigL * seededNormal();
      Ws.push(wi);
      Ls.push(li);
      areas.push(wi * li);

      // Scatter for width edge: random y in [0, L], x = sampled W
      edgeWx.push(wi);
      edgeWy.push(seededRandom() * L);

      // Scatter for length edge: random x in [0, W], y = sampled L
      edgeLx.push(seededRandom() * W);
      edgeLy.push(li);
    }

    // Statistics for area
    const areaMean = areas.reduce((a, b) => a + b, 0) / nSim;
    const areaStd = Math.sqrt(areas.reduce((a, b) => a + (b - areaMean) ** 2, 0) / (nSim - 1));

    // Analytical error propagation: sigma_A = sqrt((L*sigW)^2 + (W*sigL)^2)
    const sigAAnalytical = Math.sqrt((L * sigW) ** 2 + (W * sigL) ** 2);

    return { edgeWx, edgeWy, edgeLx, edgeLy, areas, areaMean, areaStd, sigAAnalytical };
  }, [W, L, sigW, sigL]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Error Propagation: Area of a Rectangle</h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        Consider a rectangle with width W and length L, each measured with Gaussian uncertainty.
        The area A = W * L inherits the propagated error. The left plot shows the uncertain edges
        of the rectangle; the right plot shows the resulting distribution of the area.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="text-sm text-[var(--text-muted)]">W: {W}</label>
          <Slider min={1} max={10} step={1} value={[W]}
            onValueChange={([v]) => setW(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">sigma_W: {sigW.toFixed(2)}</label>
          <Slider min={0.01} max={1.5} step={0.01} value={[sigW]}
            onValueChange={([v]) => setSigW(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">L: {L}</label>
          <Slider min={1} max={10} step={1} value={[L]}
            onValueChange={([v]) => setL(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">sigma_L: {sigL.toFixed(2)}</label>
          <Slider min={0.01} max={1.5} step={0.01} value={[sigL]}
            onValueChange={([v]) => setSigL(v)} />
        </div>
      </div>

      <div className="text-sm text-[var(--text-muted)] mb-2">
        Simulated: A = {result.areaMean.toFixed(2)} +/- {result.areaStd.toFixed(3)} |
        Analytical sigma_A = {result.sigAAnalytical.toFixed(3)}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CanvasChart
          data={[
            {
              x: result.edgeWx,
              y: result.edgeWy,
              type: 'scatter',
              mode: 'markers',
              marker: { color: 'rgba(250,204,21,0.25)', size: 3 },
              name: 'Width edge',
            },
            {
              x: result.edgeLx,
              y: result.edgeLy,
              type: 'scatter',
              mode: 'markers',
              marker: { color: 'rgba(250,204,21,0.25)', size: 3 },
              name: 'Length edge',
            },
            {
              x: [0, W, W, 0, 0],
              y: [0, 0, L, L, 0],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#fff', width: 1.5 },
              name: 'Nominal rect',
            },
          ]}
          layout={{
            title: { text: 'Rectangle with Uncertain Edges' },
            margin: { t: 40, r: 20, b: 50, l: 50 },
            xaxis: { title: { text: 'Width' } },
            yaxis: { title: { text: 'Length' } },
            legend: {},
          }}
          style={{ width: '100%', height: 400 }}
        />
        <CanvasChart
          data={[{
            x: result.areas,
            type: 'histogram',
            nbinsx: 50,
            marker: { color: 'rgba(59,130,246,0.6)' },
            name: 'Area distribution',
          }]}
          layout={{
            title: { text: 'Area Distribution' },
            margin: { t: 40, r: 20, b: 50, l: 50 },
            xaxis: { title: { text: 'Area' } },
            yaxis: { title: { text: 'Count' } },
            showlegend: false,
          }}
          style={{ width: '100%', height: 400 }}
        />
      </div>
    </div>
  );
}
