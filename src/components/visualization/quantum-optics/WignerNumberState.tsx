"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import type { SimulationComponentProps } from '@/shared/types/simulation';


/**
 * Laguerre polynomial L_n(x) computed via the recurrence relation:
 *   L_0(x) = 1
 *   L_1(x) = 1 - x
 *   L_{n+1}(x) = ((2n + 1 - x) * L_n(x) - n * L_{n-1}(x)) / (n + 1)
 */
function laguerre(n: number, x: number): number {
  if (n === 0) return 1;
  if (n === 1) return 1 - x;
  let lPrev2 = 1;
  let lPrev1 = 1 - x;
  let lCurr = 0;
  for (let k = 1; k < n; k++) {
    lCurr = ((2 * k + 1 - x) * lPrev1 - k * lPrev2) / (k + 1);
    lPrev2 = lPrev1;
    lPrev1 = lCurr;
  }
  return lCurr;
}

/**
 * Wigner function for a Fock (number) state |n>
 *
 * W(x, p) = ((-1)^n / pi) * exp(-(x^2 + p^2)) * L_n(2*(x^2 + p^2))
 *
 * where L_n is the n-th Laguerre polynomial.
 */
export default function WignerNumberState({}: SimulationComponentProps) {
  const [photonNumber, setPhotonNumber] = useState(0);
  const [show3D, setShow3D] = useState(false);

  const { xvec, pvec, W } = useMemo(() => {
    const plotRange = 5;
    const N = 140;
    const xvec = Array.from({ length: N }, (_, i) => -plotRange + (2 * plotRange * i) / (N - 1));
    const pvec = Array.from({ length: N }, (_, i) => -plotRange + (2 * plotRange * i) / (N - 1));

    const n = photonNumber;
    const sign = n % 2 === 0 ? 1 : -1;

    const W: number[][] = [];
    for (let j = 0; j < N; j++) {
      const row: number[] = [];
      for (let i = 0; i < N; i++) {
        const x = xvec[i];
        const p = pvec[j];
        const r2 = x * x + p * p;
        const val = (sign / Math.PI) * Math.exp(-r2) * laguerre(n, 2 * r2);
        row.push(val);
      }
      W.push(row);
    }
    return { xvec, pvec, W };
  }, [photonNumber]);

  return (
    <SimulationPanel title="Number State (Fock State) Wigner Function" caption="|ψ⟩ = |n⟩. The Wigner function involves Laguerre polynomials and exhibits negativity for n ≥ 1, demonstrating non-classical behavior.">
      <SimulationSettings>
        <SimulationToggle
          options={[
            { label: '2D Contour', value: '2d' },
            { label: '3D Surface', value: '3d' },
          ]}
          value={show3D ? '3d' : '2d'}
          onChange={(v) => setShow3D(v === '3d')}
        />
      </SimulationSettings>

      <SimulationConfig>
        <div>
          <SimulationLabel>
            Photon number n: {photonNumber}
          </SimulationLabel>
          <Slider
            value={[photonNumber]}
            onValueChange={(val) => setPhotonNumber(Math.round(val[0]))}
            min={0}
            max={15}
            step={1}
            className="w-full max-w-md"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
        {!show3D ? (
          <CanvasHeatmap
            data={[
              {
                z: W,
                x: xvec,
                y: pvec,
                type: 'heatmap',
                colorscale: 'RdBu',
                colorbar: { title: { text: 'W(q, p)' } },
              },
            ]}
            layout={{
              title: { text: `Fock State |${photonNumber}> W(q, p)` },
              xaxis: {
                title: { text: 'q' },
              },
              yaxis: {
                title: { text: 'p' },
                scaleanchor: 'x',
              },
              height: 500,
              margin: { t: 40, r: 20, b: 50, l: 50 },
            }}
            style={{ width: '100%', height: '500px' }}
          />
        ) : (
          <CanvasHeatmap
            data={[
              {
                z: W,
                x: xvec,
                y: pvec,
                type: 'heatmap',
                colorscale: 'RdBu',
                colorbar: { title: { text: 'W(q,p)' } },
              },
            ]}
            layout={{
              title: { text: `Fock State |${photonNumber}> W(q, p) - 3D` },
              xaxis: { title: { text: 'q' } },
              yaxis: { title: { text: 'p' } },
              height: 500,
              margin: { t: 40, r: 20, b: 50, l: 50 },
            }}
            style={{ width: '100%', height: '500px' }}
          />
        )}
      </SimulationMain>
    </SimulationPanel>
  );
}
