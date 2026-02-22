"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function StressTensor({}: SimulationComponentProps) {
  const [sigmaXX, setSigmaXX] = useState(100); // MPa
  const [sigmaYY, setSigmaYY] = useState(-50);
  const [tauXY, setTauXY] = useState(40);

  const mohrData = useMemo(() => {
    const sx = sigmaXX;
    const sy = sigmaYY;
    const txy = tauXY;

    // Center and radius of Mohr's circle
    const center = (sx + sy) / 2;
    const R = Math.sqrt(((sx - sy) / 2) ** 2 + txy ** 2);

    // Principal stresses
    const sigma1 = center + R;
    const sigma2 = center - R;

    // Principal angle (angle from x-axis to sigma1 direction)
    const theta_p = (0.5 * Math.atan2(2 * txy, sx - sy)) * (180 / Math.PI);

    // Maximum shear stress
    const tauMax = R;

    // Generate circle points
    const nPts = 200;
    const circleNormal: number[] = [];
    const circleShear: number[] = [];
    for (let i = 0; i <= nPts; i++) {
      const angle = (2 * Math.PI * i) / nPts;
      circleNormal.push(center + R * Math.cos(angle));
      circleShear.push(R * Math.sin(angle));
    }

    // Current state points: (sx, txy) and (sy, -txy)
    // These are the two points on the circle corresponding to the given orientation
    const pointANormal = sx;
    const pointAShear = txy;
    const pointBNormal = sy;
    const pointBShear = -txy;

    return {
      center,
      R,
      sigma1,
      sigma2,
      theta_p,
      tauMax,
      circleNormal,
      circleShear,
      pointANormal,
      pointAShear,
      pointBNormal,
      pointBShear,
    };
  }, [sigmaXX, sigmaYY, tauXY]);

  const plotData = useMemo(() => {
    const {
      center,
      sigma1,
      sigma2,
      tauMax,
      circleNormal,
      circleShear,
      pointANormal,
      pointAShear,
      pointBNormal,
      pointBShear,
    } = mohrData;

    // Axis padding
    const pad = Math.max(Math.abs(sigma1), Math.abs(sigma2), tauMax) * 0.3 + 20;
    const axMin = Math.min(sigma2, 0) - pad;
    const axMax = sigma1 + pad;
    const shearRange = tauMax + pad;

    return {
      data: [
        // Mohr's circle
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: circleNormal,
          y: circleShear,
          name: "Mohr's Circle",
          line: { color: '#3b82f6', width: 2 },
          hoverinfo: 'skip' as const,
        },
        // Diameter line connecting A and B
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: [pointANormal, pointBNormal],
          y: [pointAShear, pointBShear],
          name: 'Diameter (A-B)',
          line: { color: '#9ca3af', width: 1, dash: 'dot' as const },
          hoverinfo: 'skip' as const,
        },
        // Point A: (sigma_xx, tau_xy)
        {
          type: 'scatter' as const,
          mode: 'markers+text' as any,
          x: [pointANormal],
          y: [pointAShear],
          name: `A (\u03c3_xx, \u03c4_xy)`,
          marker: { color: '#ef4444', size: 10 },
          text: ['A'],
          textposition: 'top right' as const,
          textfont: { color: '#ef4444', size: 13 },
        },
        // Point B: (sigma_yy, -tau_xy)
        {
          type: 'scatter' as const,
          mode: 'markers+text' as any,
          x: [pointBNormal],
          y: [pointBShear],
          name: `B (\u03c3_yy, -\u03c4_xy)`,
          marker: { color: '#f59e0b', size: 10 },
          text: ['B'],
          textposition: 'bottom left' as const,
          textfont: { color: '#f59e0b', size: 13 },
        },
        // Principal stresses on the normal axis
        {
          type: 'scatter' as const,
          mode: 'markers+text' as any,
          x: [sigma1, sigma2],
          y: [0, 0],
          name: 'Principal stresses',
          marker: { color: '#10b981', size: 10, symbol: 'diamond' },
          text: ['\u03c3\u2081', '\u03c3\u2082'],
          textposition: 'top center' as const,
          textfont: { color: '#10b981', size: 13 },
        },
        // Center
        {
          type: 'scatter' as const,
          mode: 'markers+text' as any,
          x: [center],
          y: [0],
          name: `Center (${center.toFixed(1)})`,
          marker: { color: '#8b5cf6', size: 8, symbol: 'cross' },
          text: ['C'],
          textposition: 'bottom center' as const,
          textfont: { color: '#8b5cf6', size: 12 },
        },
        // Max shear stress markers
        {
          type: 'scatter' as const,
          mode: 'markers' as const,
          x: [center, center],
          y: [mohrData.tauMax, -mohrData.tauMax],
          name: `\u03c4_max = ${mohrData.tauMax.toFixed(1)} MPa`,
          marker: { color: '#ec4899', size: 8, symbol: 'triangle-up' },
        },
      ],
      layout: ({
        title: { text: "Mohr's Circle for 2D Stress State" },
        xaxis: {
          title: { text: 'Normal Stress \u03c3 [MPa]' },
          zerolinewidth: 1,
          range: [axMin, axMax],
          scaleanchor: 'y' as const,
          scaleratio: 1,
        },
        yaxis: {
          title: { text: 'Shear Stress \u03c4 [MPa]' },
          zerolinewidth: 1,
          range: [-shearRange, shearRange],
        },
        height: 550,
        legend: {
          bgcolor: 'rgba(0,0,0,0)',
          x: 1.02,
          y: 1,
        },
        margin: { t: 50, b: 60, l: 70, r: 180 },
      }),
    };
  }, [mohrData]);

  return (
    <SimulationPanel title="2D Mohr's Circle Visualization" caption="Enter the components of a 2D stress tensor to visualize the corresponding Mohr's circle showing principal stresses, maximum shear stress, and the current stress state.">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            &sigma;<sub>xx</sub>: {sigmaXX} MPa
          </SimulationLabel>
          <Slider
            min={-200}
            max={200}
            step={5}
            value={[sigmaXX]}
            onValueChange={([v]) => setSigmaXX(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>
            &sigma;<sub>yy</sub>: {sigmaYY} MPa
          </SimulationLabel>
          <Slider
            min={-200}
            max={200}
            step={5}
            value={[sigmaYY]}
            onValueChange={([v]) => setSigmaYY(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>
            &tau;<sub>xy</sub>: {tauXY} MPa
          </SimulationLabel>
          <Slider
            min={-150}
            max={150}
            step={5}
            value={[tauXY]}
            onValueChange={([v]) => setTauXY(v)}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasChart
          data={plotData.data}
          layout={plotData.layout}
          style={{ width: '100%', height: 550 }}
        />
      </SimulationMain>
      <SimulationResults>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-[var(--surface-2)] rounded-lg p-3 text-center">
            <div className="text-xs text-[var(--text-soft)] mb-1">Principal &sigma;<sub>1</sub></div>
            <div className="text-lg font-mono text-green-400">
              {mohrData.sigma1.toFixed(1)} MPa
            </div>
          </div>
          <div className="bg-[var(--surface-2)] rounded-lg p-3 text-center">
            <div className="text-xs text-[var(--text-soft)] mb-1">Principal &sigma;<sub>2</sub></div>
            <div className="text-lg font-mono text-green-400">
              {mohrData.sigma2.toFixed(1)} MPa
            </div>
          </div>
          <div className="bg-[var(--surface-2)] rounded-lg p-3 text-center">
            <div className="text-xs text-[var(--text-soft)] mb-1">Max Shear &tau;<sub>max</sub></div>
            <div className="text-lg font-mono text-pink-400">
              {mohrData.tauMax.toFixed(1)} MPa
            </div>
          </div>
          <div className="bg-[var(--surface-2)] rounded-lg p-3 text-center">
            <div className="text-xs text-[var(--text-soft)] mb-1">Principal Angle &theta;<sub>p</sub></div>
            <div className="text-lg font-mono text-purple-400">
              {mohrData.theta_p.toFixed(1)}&deg;
            </div>
          </div>
        </div>
      </SimulationResults>

      <div className="mt-4 text-sm text-[var(--text-muted)] space-y-1">
        <p>
          <strong className="text-[var(--text-muted)]">Point A</strong> represents the stress
          state on the plane with its normal along x: (&sigma;<sub>xx</sub>, &tau;<sub>xy</sub>).
        </p>
        <p>
          <strong className="text-[var(--text-muted)]">Point B</strong> represents the plane
          with its normal along y: (&sigma;<sub>yy</sub>, &minus;&tau;<sub>xy</sub>).
        </p>
        <p>
          The <strong className="text-[var(--text-muted)]">green diamonds</strong> mark the
          principal stresses where shear vanishes. The principal angle &theta;<sub>p</sub> is
          measured from the x-axis to the direction of &sigma;<sub>1</sub>.
        </p>
      </div>
    </SimulationPanel>
  );
}
