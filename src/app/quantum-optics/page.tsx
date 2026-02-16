'use client';

import dynamic from 'next/dynamic';
import { useState } from 'react';
import { create, all } from 'mathjs';
import type { Complex } from 'mathjs';

const math = create(all);
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function QuantumOptics() {
  const [theta1, setTheta1] = useState(0);
  const [phi1, setPhi1] = useState(0);
  const [epsilon1, setEpsilon1] = useState(0);
  const [theta2, setTheta2] = useState(Math.PI / 4);
  const [phi2, setPhi2] = useState(Math.PI / 2);
  const [epsilon2, setEpsilon2] = useState(Math.PI / 4);
  const [delta, setDelta] = useState(0);

  // Function to compute Jones vector
  const computeJones = (theta: number, phi: number, epsilon: number) => {
    const cosEps = math.cos(epsilon / 2);
    const sinEps = math.multiply(math.sin(epsilon / 2), math.exp(math.multiply(math.i, phi)));
    const base = [cosEps, sinEps];
    const rot = [
      [math.cos(theta), -math.sin(theta)],
      [math.sin(theta), math.cos(theta)]
    ];
    return [
      math.add(math.multiply(rot[0][0], base[0]), math.multiply(rot[0][1], base[1])),
      math.add(math.multiply(rot[1][0], base[0]), math.multiply(rot[1][1], base[1]))
    ];
  };

  const J1 = computeJones(theta1, phi1, epsilon1);
  const J2 = computeJones(theta2, phi2, epsilon2);
  const phase = math.exp(math.multiply(math.i, delta));
  const J2Phased = [math.multiply(J2[0], phase), math.multiply(J2[1], phase)];
  const JTotal = [math.add(J1[0], J2Phased[0]), math.add(J1[1], J2Phased[1])];

  // Normalize to unit intensity
  const norm1 = math.sqrt(math.add(math.multiply(math.conj(J1[0]), J1[0]), math.multiply(math.conj(J1[1]), J1[1])));
  const norm2 = math.sqrt(math.add(math.multiply(math.conj(J2[0]), J2[0]), math.multiply(math.conj(J2[1]), J2[1])));
  const normTotal = math.sqrt(math.add(math.multiply(math.conj(JTotal[0]), JTotal[0]), math.multiply(math.conj(JTotal[1]), JTotal[1])));
  const J1Norm = [math.divide(J1[0], norm1), math.divide(J1[1], norm1)];
  const J2Norm = [math.divide(J2[0], norm2), math.divide(J2[1], norm2)];
  const JTotalNorm = normTotal ? [math.divide(JTotal[0], normTotal), math.divide(JTotal[1], normTotal)] : [0, 0];

  // Generate polarization ellipse
  const generateEllipse = (J: Complex[]) => {
    const x = [];
    const y = [];
    for (let t = 0; t < 2 * Math.PI; t += 0.1) {
      const phase = math.exp(math.multiply(math.i, -t));
      const Ex = math.re(math.multiply(J[0], phase));
      const Ey = math.re(math.multiply(J[1], phase));
      x.push(Ex);
      y.push(Ey);
    }
    return { x, y };
  };

  const ellipse1 = generateEllipse(J1Norm);
  const ellipse2 = generateEllipse(J2Norm);
  const ellipseTotal = generateEllipse(JTotalNorm);

  // Data for plots
  const data1 = [
    {
      x: ellipse1.x,
      y: ellipse1.y,
      type: 'scatter',
      mode: 'lines',
      name: 'Beam 1',
      line: { color: 'blue' },
    },
  ];

  const data2 = [
    {
      x: ellipse2.x,
      y: ellipse2.y,
      type: 'scatter',
      mode: 'lines',
      name: 'Beam 2',
      line: { color: 'red' },
    },
  ];

  const dataTotal = [
    {
      x: ellipseTotal.x,
      y: ellipseTotal.y,
      type: 'scatter',
      mode: 'lines',
      name: 'Total',
      line: { color: 'green' },
    },
  ];

  const layout = {
    title: 'Polarization Ellipse',
    xaxis: { title: 'E_x', scaleanchor: 'y' },
    yaxis: { title: 'E_y', scaleanchor: 'x' },
    showlegend: true,
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">Photon Polarization and Interference</h1>
      <p className="mb-4">
        This interactive demonstration visualizes Jones vectors and polarization states for photons.
        Adjust the parameters for two beams and their relative phase to see interference effects.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div>
          <h2 className="text-xl font-semibold mb-2">Beam 1</h2>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Orientation θ: {(theta1 * 180 / Math.PI).toFixed(1)}°</label>
            <input
              type="range"
              min="0"
              max={2 * Math.PI}
              step="0.1"
              value={theta1}
              onChange={(e) => setTheta1(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Phase φ: {(phi1 * 180 / Math.PI).toFixed(1)}°</label>
            <input
              type="range"
              min="0"
              max={2 * Math.PI}
              step="0.1"
              value={phi1}
              onChange={(e) => setPhi1(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Ellipticity ε: {(epsilon1 * 180 / Math.PI).toFixed(1)}°</label>
            <input
              type="range"
              min="0"
              max={Math.PI / 2}
              step="0.1"
              value={epsilon1}
              onChange={(e) => setEpsilon1(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <p>Jones Vector: [{math.re(J1Norm[0]).toFixed(2)} + {math.im(J1Norm[0]).toFixed(2)}i, {math.re(J1Norm[1]).toFixed(2)} + {math.im(J1Norm[1]).toFixed(2)}i]</p>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-2">Beam 2</h2>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Orientation θ: {(theta2 * 180 / Math.PI).toFixed(1)}°</label>
            <input
              type="range"
              min="0"
              max={2 * Math.PI}
              step="0.1"
              value={theta2}
              onChange={(e) => setTheta2(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Phase φ: {(phi2 * 180 / Math.PI).toFixed(1)}°</label>
            <input
              type="range"
              min="0"
              max={2 * Math.PI}
              step="0.1"
              value={phi2}
              onChange={(e) => setPhi2(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Ellipticity ε: {(epsilon2 * 180 / Math.PI).toFixed(1)}°</label>
            <input
              type="range"
              min="0"
              max={Math.PI / 2}
              step="0.1"
              value={epsilon2}
              onChange={(e) => setEpsilon2(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <p>Jones Vector: [{math.re(J2Norm[0]).toFixed(2)} + {math.im(J2Norm[0]).toFixed(2)}i, {math.re(J2Norm[1]).toFixed(2)} + {math.im(J2Norm[1]).toFixed(2)}i]</p>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-2">Interference</h2>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Relative Phase δ: {(delta * 180 / Math.PI).toFixed(1)}°</label>
            <input
              type="range"
              min="0"
              max={2 * Math.PI}
              step="0.1"
              value={delta}
              onChange={(e) => setDelta(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <p>Total Jones Vector: [{math.re(JTotalNorm[0]).toFixed(2)} + {math.im(JTotalNorm[0]).toFixed(2)}i, {math.re(JTotalNorm[1]).toFixed(2)} + {math.im(JTotalNorm[1]).toFixed(2)}i]</p>
          <p>Intensity: {math.re(normTotal ** 2).toFixed(2)}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <Plot
            data={data1}
            layout={{ ...layout, title: 'Beam 1 Polarization' }}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
        <div>
          <Plot
            data={data2}
            layout={{ ...layout, title: 'Beam 2 Polarization' }}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
        <div>
          <Plot
            data={dataTotal}
            layout={{ ...layout, title: 'Total Polarization (Interference)' }}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
      </div>

      <div className="mt-6">
        <h2 className="text-2xl font-semibold mb-2">References</h2>
        <ul className="list-disc list-inside">
          <li>Wikipedia: Jones Calculus</li>
          <li>Quantum Optics textbooks by Scully and Zubairy</li>
        </ul>
      </div>
    </div>
  );
}