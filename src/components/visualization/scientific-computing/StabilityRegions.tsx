"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationToggle, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Stability regions for time-stepping methods in the complex plane.
 * Shows where |amplification factor| < 1 for:
 * - Forward Euler
 * - Backward Euler
 * - Crank-Nicolson
 * - RK4
 *
 * Optionally overlay eigenvalues of a test problem.
 */

type MethodKey = 'forward-euler' | 'backward-euler' | 'crank-nicolson' | 'rk4';

const METHODS: { key: MethodKey; label: string; color: string }[] = [
  { key: 'forward-euler', label: 'Forward Euler', color: '#3b82f6' },
  { key: 'backward-euler', label: 'Backward Euler', color: '#10b981' },
  { key: 'crank-nicolson', label: 'Crank-Nicolson', color: '#f59e0b' },
  { key: 'rk4', label: 'RK4', color: '#ef4444' },
];

/**
 * Amplification factor for each method with z = h*lambda:
 * Forward Euler: |1 + z|
 * Backward Euler: |1/(1 - z)|
 * Crank-Nicolson: |(1 + z/2)/(1 - z/2)|
 * RK4: |1 + z + z^2/2 + z^3/6 + z^4/24|
 */
function amplificationFactor(method: MethodKey, re: number, im: number): number {
  switch (method) {
    case 'forward-euler': {
      const rr = 1 + re;
      const ri = im;
      return Math.sqrt(rr * rr + ri * ri);
    }
    case 'backward-euler': {
      const dr = 1 - re;
      const di = -im;
      const denom = dr * dr + di * di;
      if (denom < 1e-20) return 1e10;
      return 1 / Math.sqrt(denom);
    }
    case 'crank-nicolson': {
      const nr = 1 + re / 2;
      const ni = im / 2;
      const dr = 1 - re / 2;
      const di = -im / 2;
      const numMag = Math.sqrt(nr * nr + ni * ni);
      const denMag = Math.sqrt(dr * dr + di * di);
      if (denMag < 1e-20) return 1e10;
      return numMag / denMag;
    }
    case 'rk4': {
      // R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
      // Compute z^k iteratively in complex arithmetic
      const zr = re, zi = im;
      let rr = 1 + re, ri = im; // 1 + z

      // z^2
      const z2r = zr * re - zi * im;
      const z2i = zr * im + zi * re;
      rr += z2r / 2;
      ri += z2i / 2;

      // z^3
      const z3r = z2r * re - z2i * im;
      const z3i = z2r * im + z2i * re;
      rr += z3r / 6;
      ri += z3i / 6;

      // z^4
      const z4r = z3r * re - z3i * im;
      const z4i = z3r * im + z3i * re;
      rr += z4r / 24;
      ri += z4i / 24;

      return Math.sqrt(rr * rr + ri * ri);
    }
  }
}

export default function StabilityRegions({}: SimulationComponentProps) {
  const [selectedMethod, setSelectedMethod] = useState<MethodKey>('forward-euler');
  const [showAll, setShowAll] = useState(false);
  const [resolution, setResolution] = useState(200);
  const [zoom, setZoom] = useState(4.0);
  const [showEigenvalues, setShowEigenvalues] = useState(true);
  const [stiffnessRatio, setStiffnessRatio] = useState(50);

  // Compute stability region heatmap
  const heatmapData = useMemo(() => {
    const n = resolution;
    const xMin = -zoom * 1.2;
    const xMax = zoom * 0.8;
    const yMin = -zoom;
    const yMax = zoom;

    const xGrid = Array.from({ length: n }, (_, i) => xMin + (xMax - xMin) * i / (n - 1));
    const yGrid = Array.from({ length: n }, (_, i) => yMin + (yMax - yMin) * i / (n - 1));

    const z: number[][] = [];

    if (showAll) {
      // Overlay: value encodes which methods are stable (bit mask -> sum)
      for (let iy = 0; iy < n; iy++) {
        const row: number[] = [];
        for (let ix = 0; ix < n; ix++) {
          const re = xGrid[ix];
          const im = yGrid[iy];
          let val = 0;
          for (let m = 0; m < METHODS.length; m++) {
            const amp = amplificationFactor(METHODS[m].key, re, im);
            if (amp <= 1.0) val += (m + 1);
          }
          row.push(val);
        }
        z.push(row);
      }
    } else {
      // Single method: value = amplification factor (clipped)
      for (let iy = 0; iy < n; iy++) {
        const row: number[] = [];
        for (let ix = 0; ix < n; ix++) {
          const re = xGrid[ix];
          const im = yGrid[iy];
          const amp = amplificationFactor(selectedMethod, re, im);
          // Map: <1 stable (low values), >1 unstable (high values)
          row.push(Math.min(amp, 2.0));
        }
        z.push(row);
      }
    }

    return { z, xGrid, yGrid };
  }, [selectedMethod, showAll, resolution, zoom]);

  // Test problem eigenvalues: stiff system with eigenvalues at -1 and -stiffnessRatio
  const eigenvalueInfo = useMemo(() => {
    const eigenvalues = [
      { re: -1, im: 0, label: '\u03BB1 = -1' },
      { re: -stiffnessRatio, im: 0, label: `\u03BB2 = -${stiffnessRatio}` },
      { re: -stiffnessRatio / 4, im: stiffnessRatio / 3, label: '\u03BB3' },
      { re: -stiffnessRatio / 4, im: -stiffnessRatio / 3, label: '\u03BB4' },
    ];
    return eigenvalues;
  }, [stiffnessRatio]);

  const methodInfo = METHODS.find((m) => m.key === selectedMethod)!;

  return (
    <SimulationPanel title="Stability Regions of Time-Stepping Methods" caption="The stability region is where |amplification factor| \u2264 1 in the complex h\u03BB plane. Explicit methods have bounded regions; implicit methods cover the left half-plane.">
      <SimulationSettings>
        <div>
          <SimulationLabel>Method</SimulationLabel>
          <SimulationToggle
            options={METHODS.map((m) => ({ label: m.label.split(' ').pop()!, value: m.key }))}
            value={selectedMethod}
            onChange={(v) => { setSelectedMethod(v as MethodKey); setShowAll(false); }}
          />
        </div>
        <div className="flex items-center gap-2">
          <SimulationButton
            variant="secondary"
            onClick={() => setShowAll(!showAll)}
          >
            Show All
          </SimulationButton>
          <SimulationButton
            variant="secondary"
            onClick={() => setShowEigenvalues(!showEigenvalues)}
          >
            {showEigenvalues ? 'Hide' : 'Show'} Test Eigenvalues
          </SimulationButton>
          {showEigenvalues && (
            <span className="text-xs text-[var(--text-muted)]">
              Stiff system: {'\u03BB'} = -1, -{stiffnessRatio}, -{stiffnessRatio / 4} &plusmn; {(stiffnessRatio / 3).toFixed(0)}i
            </span>
          )}
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <SimulationLabel>Resolution: {resolution}</SimulationLabel>
            <Slider
              value={[resolution]}
              onValueChange={([v]) => setResolution(v)}
              min={80}
              max={400}
              step={20}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Zoom: {zoom.toFixed(1)}</SimulationLabel>
            <Slider
              value={[zoom]}
              onValueChange={([v]) => setZoom(v)}
              min={1}
              max={8}
              step={0.5}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Stiffness ratio: {stiffnessRatio}</SimulationLabel>
            <Slider
              value={[stiffnessRatio]}
              onValueChange={([v]) => setStiffnessRatio(v)}
              min={2}
              max={200}
              step={1}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasHeatmap
          data={[
            {
              z: heatmapData.z,
              x: heatmapData.xGrid,
              y: heatmapData.yGrid,
              colorscale: showAll ? 'Portland' : 'RdBu',
              showscale: true,
              reversescale: !showAll,
              zmin: showAll ? 0 : 0,
              zmax: showAll ? 10 : 2,
            },
          ]}
          layout={{
            title: {
              text: showAll
                ? 'Stability Regions (all methods overlaid)'
                : `|g(h\u03BB)| for ${methodInfo.label} (blue = stable)`,
            },
            xaxis: { title: { text: 'Re(h\u03BB)' } },
            yaxis: { title: { text: 'Im(h\u03BB)' } },
            margin: { t: 40, r: 60, b: 50, l: 60 },
            ...(showEigenvalues
              ? {
                  annotations: eigenvalueInfo.map((ev) => ({
                    x: ev.re,
                    y: ev.im,
                    text: '\u2716',
                    showarrow: false,
                    font: { size: 14, color: '#ffffff' },
                  })),
                }
              : {}),
          }}
          style={{ width: '100%', height: 450 }}
        />
      </SimulationMain>

      <SimulationResults>
        <div className="text-xs text-[var(--text-muted)]">
          <strong>Key insight:</strong> For the selected stiffness ratio of {stiffnessRatio},
          Forward Euler needs h &lt; {(2 / stiffnessRatio).toFixed(4)} to be stable,
          while Backward Euler and Crank-Nicolson are stable for any h.
          RK4 has a larger but still bounded stability region.
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
