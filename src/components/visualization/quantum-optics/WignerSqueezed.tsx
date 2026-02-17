'use client';

import dynamic from 'next/dynamic';
import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id?: string;
}

/**
 * Wigner function for a squeezed coherent state |alpha, xi>
 *
 * The squeezed state is D(alpha) S(xi) |0> where:
 *   D(alpha) is the displacement operator
 *   S(xi) is the squeeze operator with xi = r * exp(i*theta)
 *
 * The Wigner function is a rotated, anisotropic Gaussian:
 *
 *   W(x, p) = (1/pi) * exp(
 *     -e^{2r} * ((x - x0)*cos(theta/2) + (p - p0)*sin(theta/2))^2
 *     -e^{-2r} * (-(x - x0)*sin(theta/2) + (p - p0)*cos(theta/2))^2
 *   )
 *
 * where alpha = (x0 + i*p0)/sqrt(2)
 */
export default function WignerSqueezed({}: SimulationProps) {
  const [realAlpha, setRealAlpha] = useState(0.0);
  const [imagAlpha, setImagAlpha] = useState(0.0);
  const [squeezeR, setSqueezeR] = useState(0.8);
  const [squeezeTheta, setSqueezeTheta] = useState(0.0);
  const [show3D, setShow3D] = useState(false);

  const { xvec, pvec, W } = useMemo(() => {
    const plotRange = 6;
    const N = 120;
    const xvec = Array.from({ length: N }, (_, i) => -plotRange + (2 * plotRange * i) / (N - 1));
    const pvec = Array.from({ length: N }, (_, i) => -plotRange + (2 * plotRange * i) / (N - 1));

    const x0 = Math.sqrt(2) * realAlpha;
    const p0 = Math.sqrt(2) * imagAlpha;
    const halfTheta = squeezeTheta / 2;
    const cosHT = Math.cos(halfTheta);
    const sinHT = Math.sin(halfTheta);
    const exp2r = Math.exp(2 * squeezeR);
    const expNeg2r = Math.exp(-2 * squeezeR);

    const W: number[][] = [];
    for (let j = 0; j < N; j++) {
      const row: number[] = [];
      for (let i = 0; i < N; i++) {
        const dx = xvec[i] - x0;
        const dp = pvec[j] - p0;
        // Rotate into squeezed frame
        const u = dx * cosHT + dp * sinHT;
        const v = -dx * sinHT + dp * cosHT;
        const exponent = -(exp2r * u * u + expNeg2r * v * v);
        const val = (1 / Math.PI) * Math.exp(exponent);
        row.push(val);
      }
      W.push(row);
    }
    return { xvec, pvec, W };
  }, [realAlpha, imagAlpha, squeezeR, squeezeTheta]);

  const darkLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#9ca3af', family: 'system-ui' },
    margin: { t: 40, r: 20, b: 50, l: 50 },
  };

  return (
    <div className="space-y-4 bg-[#151525] rounded-lg p-4 border border-[#2d2d44]">
      <h3 className="text-lg font-semibold text-white">Squeezed State Wigner Function</h3>
      <p className="text-sm text-gray-400">
        {"|psi⟩ = |α, ξ⟩ = D(α) S(ξ)|0⟩, where S(ξ) = exp[½(ξ* â² − ξ (â†)²)] and ξ = r·exp(iθ). Squeezing reduces fluctuations in one quadrature at the expense of the other."}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-gray-300 mb-1 block">
            Re(alpha): {realAlpha.toFixed(2)}
          </label>
          <Slider
            value={[realAlpha]}
            onValueChange={(val) => setRealAlpha(val[0])}
            min={-3}
            max={3}
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
            Squeeze parameter r: {squeezeR.toFixed(2)}
          </label>
          <Slider
            value={[squeezeR]}
            onValueChange={(val) => setSqueezeR(val[0])}
            min={0}
            max={2.0}
            step={0.05}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-sm text-gray-300 mb-1 block">
            Squeeze angle theta: {(squeezeTheta * 180 / Math.PI).toFixed(0)} deg
          </label>
          <Slider
            value={[squeezeTheta]}
            onValueChange={(val) => setSqueezeTheta(val[0])}
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
              ncontours: 30,
              colorbar: { title: { text: 'W(q, p)' } },
            },
          ]}
          layout={{
            ...darkLayout,
            title: { text: `Squeezed State W(q, p) [r=${squeezeR.toFixed(2)}]` },
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
            title: { text: `Squeezed State W(q, p) - 3D [r=${squeezeR.toFixed(2)}]` },
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
