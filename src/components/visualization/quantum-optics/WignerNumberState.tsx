'use client';

import dynamic from 'next/dynamic';
import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id?: string;
}

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
export default function WignerNumberState({}: SimulationProps) {
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

  const darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#9ca3af', family: 'system-ui' },
    margin: { t: 40, r: 20, b: 50, l: 50 },
  };

  return (
    <div className="space-y-4 bg-[#151525] rounded-lg p-4 border border-[#2d2d44]">
      <h3 className="text-lg font-semibold text-white">Number State (Fock State) Wigner Function</h3>
      <p className="text-sm text-gray-400">
        {"|ψ⟩ = |n⟩. The Wigner function involves Laguerre polynomials and exhibits negativity for n ≥ 1, demonstrating non-classical behavior."}
      </p>

      <div>
        <label className="text-sm text-gray-300 mb-1 block">
          Photon number n: {photonNumber}
        </label>
        <Slider
          value={[photonNumber]}
          onValueChange={(val) => setPhotonNumber(Math.round(val[0]))}
          min={0}
          max={15}
          step={1}
          className="w-full max-w-md"
        />
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
              ncontours: 30,
              colorbar: { title: { text: 'W(q, p)' } },
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: `Fock State |${photonNumber}> W(q, p)` },
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
              colorbar: { title: { text: 'W(q,p)' } },
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: `Fock State |${photonNumber}> W(q, p) - 3D` },
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
