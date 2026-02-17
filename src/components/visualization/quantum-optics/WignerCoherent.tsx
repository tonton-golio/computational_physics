'use client';

import dynamic from 'next/dynamic';
import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';

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

  const darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#9ca3af', family: 'system-ui' },
    margin: { t: 40, r: 20, b: 50, l: 50 },
  };

  return (
    <div className="space-y-4 bg-[#151525] rounded-lg p-4 border border-[#2d2d44]">
      <h3 className="text-lg font-semibold text-white">Coherent State Wigner Function</h3>
      <p className="text-sm text-gray-400">
        {"|ψ⟩ = |α⟩ = D(α)|0⟩, where D(α) = exp(α·â† − α*·â). The Wigner function is a simple Gaussian centered at (x₀, p₀) in phase space."}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-gray-300 mb-1 block">
            Re(alpha): {realAlpha.toFixed(2)}
          </label>
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
          <label className="text-sm text-gray-300 mb-1 block">
            Im(alpha): {imagAlpha.toFixed(2)}
          </label>
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

      <div className="text-sm text-gray-400">
        |alpha|^2 = n_bar = {meanPhotonNumber.toFixed(2)}
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
              contours: { ncontours: 30 },
              colorbar: { title: { text: 'W(q, p)' }, titleside: 'right' as const },
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: 'Coherent State W(q, p)' },
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
            title: { text: 'Coherent State W(q, p) - 3D' },
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
