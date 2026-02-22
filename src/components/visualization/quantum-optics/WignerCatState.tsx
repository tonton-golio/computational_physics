"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import type { SimulationComponentProps } from '@/shared/types/simulation';


/**
 * Wigner function for a Schrodinger cat state:
 *   |psi> = N * (|alpha> + e^{iPhi} |-alpha>)
 *
 * where N = 1/sqrt(2 + 2*cos(Phi)*exp(-2|alpha|^2))
 *
 * The Wigner function is the sum of two coherent-state Gaussians plus
 * an interference term with oscillatory fringes:
 *
 *   W(x,p) = N^2 / pi * [
 *     exp(-(x-x0)^2 - (p-p0)^2)
 *     + exp(-(x+x0)^2 - (p+p0)^2)
 *     + 2 * exp(-x^2 - p^2) * cos(2*(p*x0 - x*p0) + Phi)
 *   ]
 *
 * where alpha = (x0 + i*p0)/sqrt(2)
 */
export default function WignerCatState({}: SimulationComponentProps) {
  const [realAlpha, setRealAlpha] = useState(2.5);
  const [imagAlpha, setImagAlpha] = useState(0.0);
  const [phi, setPhi] = useState(0.0);
  const [show3D, setShow3D] = useState(false);

  const { xvec, pvec, W } = useMemo(() => {
    const plotRange = 7;
    const N = 160;
    const xvec = Array.from({ length: N }, (_, i) => -plotRange + (2 * plotRange * i) / (N - 1));
    const pvec = Array.from({ length: N }, (_, i) => -plotRange + (2 * plotRange * i) / (N - 1));

    const x0 = Math.sqrt(2) * realAlpha;
    const p0 = Math.sqrt(2) * imagAlpha;
    const alphaSq = realAlpha * realAlpha + imagAlpha * imagAlpha;

    // Normalization: N^2 = 1 / (2 + 2*cos(phi)*exp(-2*|alpha|^2))
    const normSq = 1 / (2 + 2 * Math.cos(phi) * Math.exp(-2 * alphaSq));

    const W: number[][] = [];
    for (let j = 0; j < N; j++) {
      const row: number[] = [];
      for (let i = 0; i < N; i++) {
        const x = xvec[i];
        const p = pvec[j];

        // Two coherent state contributions
        const g1 = Math.exp(-Math.pow(x - x0, 2) - Math.pow(p - p0, 2));
        const g2 = Math.exp(-Math.pow(x + x0, 2) - Math.pow(p + p0, 2));

        // Interference term
        const gInt = Math.exp(-x * x - p * p);
        const interPhase = 2 * (p * x0 - x * p0) + phi;
        const interference = 2 * gInt * Math.cos(interPhase);

        const val = (normSq / Math.PI) * (g1 + g2 + interference);
        row.push(val);
      }
      W.push(row);
    }
    return { xvec, pvec, W };
  }, [realAlpha, imagAlpha, phi]);

  const catType = phi === 0 ? 'even cat' : Math.abs(phi - Math.PI) < 0.1 ? 'odd cat' : 'general superposition';

  return (
    <SimulationPanel title="Schrodinger Cat State Wigner Function" caption="|ψ⟩ = N(|α⟩ + e^(iΦ)|−α⟩). The interference fringes between the two coherent state components are a hallmark of quantum superposition. Φ=0 gives the even cat state; Φ=π gives the odd cat state.">
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
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <SimulationLabel>
              Re(alpha): {realAlpha.toFixed(2)}
            </SimulationLabel>
            <Slider
              value={[realAlpha]}
              onValueChange={(val) => setRealAlpha(val[0])}
              min={0}
              max={4}
              step={0.1}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>
              Im(alpha): {imagAlpha.toFixed(2)}
            </SimulationLabel>
            <Slider
              value={[imagAlpha]}
              onValueChange={(val) => setImagAlpha(val[0])}
              min={-3}
              max={3}
              step={0.1}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>
              Phi: {(phi * 180 / Math.PI).toFixed(0)} deg ({catType})
            </SimulationLabel>
            <Slider
              value={[phi]}
              onValueChange={(val) => setPhi(val[0])}
              min={0}
              max={2 * Math.PI}
              step={0.05}
              className="w-full"
            />
          </div>
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
              title: { text: `Cat State W(q, p)` },
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
              title: { text: `Cat State W(q, p) - 3D` },
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
