'use client';

import dynamic from 'next/dynamic';
import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id?: string;
}

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
export default function WignerCatState({}: SimulationProps) {
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

  const darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#9ca3af', family: 'system-ui' },
    margin: { t: 40, r: 20, b: 50, l: 50 },
  };

  return (
    <div className="space-y-4 bg-[#151525] rounded-lg p-4 border border-[#2d2d44]">
      <h3 className="text-lg font-semibold text-white">Schrodinger Cat State Wigner Function</h3>
      <p className="text-sm text-gray-400">
        {"|ψ⟩ = N(|α⟩ + e^(iΦ)|−α⟩). The interference fringes between the two coherent state components are a hallmark of quantum superposition. Φ=0 gives the even cat state; Φ=π gives the odd cat state."}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="text-sm text-gray-300 mb-1 block">
            Re(alpha): {realAlpha.toFixed(2)}
          </label>
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
          <label className="text-sm text-gray-300 mb-1 block">
            Im(alpha): {imagAlpha.toFixed(2)}
          </label>
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
          <label className="text-sm text-gray-300 mb-1 block">
            Phi: {(phi * 180 / Math.PI).toFixed(0)} deg ({catType})
          </label>
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

      <div className="flex gap-2">
        <button
          onClick={() => setShow3D(false)}
          className={`px-3 py-1 rounded text-sm ${!show3D ? 'bg-blue-600 text-white' : 'bg-[#1e1e2e] text-gray-400'}`}
        >
          2D Contour
        </button>
        <button
          onClick={() => setShow3D(true)}
          className={`px-3 py-1 rounded text-sm ${show3D ? 'bg-blue-600 text-white' : 'bg-[#1e1e2e] text-gray-400'}`}
        >
          3D Surface
        </button>
      </div>

      {!show3D ? (
        <Plot
          data={[
            {
              z: W,
              x: xvec,
              y: pvec,
              type: 'contour',
              colorscale: 'RdBu',
              zmid: 0,
              contours: { ncontours: 40 },
              colorbar: { title: { text: 'W(q, p)' }, titleside: 'right' as const },
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: `Cat State W(q, p)` },
            xaxis: {
              title: { text: 'q' },
              gridcolor: '#1e1e2e',
              zerolinecolor: '#2d2d44',
            },
            yaxis: {
              title: { text: 'p' },
              gridcolor: '#1e1e2e',
              zerolinecolor: '#2d2d44',
              scaleanchor: 'x',
            },
            width: undefined,
            height: 500,
          }}
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
              cmid: 0,
              colorbar: { title: { text: 'W(q,p)' }, titleside: 'right' as const },
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: `Cat State W(q, p) - 3D` },
            scene: {
              xaxis: { title: { text: 'q' }, gridcolor: '#1e1e2e' },
              yaxis: { title: { text: 'p' }, gridcolor: '#1e1e2e' },
              zaxis: { title: { text: 'W(q,p)' }, gridcolor: '#1e1e2e' },
              bgcolor: 'rgba(15,15,25,1)',
            },
            width: undefined,
            height: 500,
          }}
          style={{ width: '100%', height: '500px' }}
          config={{ responsive: true, displayModeBar: false }}
        />
      )}
    </div>
  );
}
