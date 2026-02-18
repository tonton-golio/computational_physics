'use client';

import dynamic from 'next/dynamic';
import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';
import { SimulationPanel, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id?: string;
}

/**
 * Wigner function for a coherent state |alpha>
 *
 * W(x, p) = (1/pi) * exp(-(x - x0)^2 - (p - p0)^2)
 *
 * where alpha = (x0 + i*p0) / sqrt(2), so x0 = sqrt(2)*Re(alpha), p0 = sqrt(2)*Im(alpha)
 */
export default function WignerCoherent({}: SimulationProps) {
  const [realAlpha, setRealAlpha] = useState(2.0);
  const [imagAlpha, setImagAlpha] = useState(0.0);
  const [show3D, setShow3D] = useState(false);
  const { mergeLayout } = usePlotlyTheme();

  const { xvec, pvec, W } = useMemo(() => {
    const plotRange = 6;
    const N = 120;
    const xvec = Array.from({ length: N }, (_, i) => -plotRange + (2 * plotRange * i) / (N - 1));
    const pvec = Array.from({ length: N }, (_, i) => -plotRange + (2 * plotRange * i) / (N - 1));

    // For coherent state: alpha = (x0 + i*p0)/sqrt(2)
    // So x0 = sqrt(2) * Re(alpha), p0 = sqrt(2) * Im(alpha)
    const x0 = Math.sqrt(2) * realAlpha;
    const p0 = Math.sqrt(2) * imagAlpha;

    const W: number[][] = [];
    for (let j = 0; j < N; j++) {
      const row: number[] = [];
      for (let i = 0; i < N; i++) {
        const x = xvec[i];
        const p = pvec[j];
        const val = (1 / Math.PI) * Math.exp(-Math.pow(x - x0, 2) - Math.pow(p - p0, 2));
        row.push(val);
      }
      W.push(row);
    }
    return { xvec, pvec, W };
  }, [realAlpha, imagAlpha]);

  const meanPhotonNumber = realAlpha ** 2 + imagAlpha ** 2;

  return (
    <SimulationPanel>
      <h3 className="text-lg font-semibold text-[var(--text-strong)]">Coherent State Wigner Function</h3>
      <p className="text-sm text-[var(--text-soft)]">
        {"|ψ⟩ = |α⟩ = D(α)|0⟩, where D(α) = exp(α·â† − α*·â). The Wigner function is a simple Gaussian centered at (x₀, p₀) in phase space."}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <SimulationLabel>
            Re(alpha): {realAlpha.toFixed(2)}
          </SimulationLabel>
          <Slider
            value={[realAlpha]}
            onValueChange={(val) => setRealAlpha(val[0])}
            min={-4}
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
            min={-4}
            max={4}
            step={0.1}
            className="w-full"
          />
        </div>
      </div>

      <div className="text-sm text-[var(--text-muted)]">
        |alpha|^2 = n_bar = {meanPhotonNumber.toFixed(2)}
      </div>

      <SimulationToggle
        options={[
          { label: '2D Contour', value: '2d' },
          { label: '3D Surface', value: '3d' },
        ]}
        value={show3D ? '3d' : '2d'}
        onChange={(v) => setShow3D(v === '3d')}
      />

      {!show3D ? (
        <Plot
          data={[
            {
              z: W,
              x: xvec,
              y: pvec,
              type: 'contour',
              colorscale: 'RdBu',
              ncontours: 30,
              colorbar: { title: { text: 'W(q, p)' } },
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Coherent State W(q, p)' },
            xaxis: {
              title: { text: 'q' },
            },
            yaxis: {
              title: { text: 'p' },
              scaleanchor: 'x',
            },
            height: 500,
            margin: { t: 40, r: 20, b: 50, l: 50 },
          })}
          style={{ width: '100%', height: '500px' }}
          config={{ responsive: true, displayModeBar: false }}
        />
      ) : (
        <Plot
          data={[
            {
              z: W,
              x: xvec,
              y: pvec,
              type: 'surface',
              colorscale: 'RdBu',
              colorbar: { title: { text: 'W(q,p)' } },
            },
          ]}
          layout={mergeLayout({
            title: { text: 'Coherent State W(q, p) - 3D' },
            scene: {
              xaxis: { title: { text: 'q' } },
              yaxis: { title: { text: 'p' } },
              zaxis: { title: { text: 'W(q,p)' } },
            },
            height: 500,
            margin: { t: 40, r: 20, b: 50, l: 50 },
          })}
          style={{ width: '100%', height: '500px' }}
          config={{ responsive: true, displayModeBar: false }}
        />
      )}
    </SimulationPanel>
  );
}
