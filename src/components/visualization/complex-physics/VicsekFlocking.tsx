'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';

const BOX_SIZE = 25;
const RADIUS = 1;
const STEPS = 200;

function runVicsek(N: number, eta: number, v0: number) {
  // Initialize particles
  const x = new Float64Array(N);
  const y = new Float64Array(N);
  const theta = new Float64Array(N);

  for (let i = 0; i < N; i++) {
    x[i] = Math.random() * BOX_SIZE;
    y[i] = Math.random() * BOX_SIZE;
    theta[i] = Math.random() * 2 * Math.PI;
  }

  const orderParams: number[] = [];

  // Run simulation
  for (let step = 0; step < STEPS; step++) {
    // Compute order parameter
    let sumCos = 0;
    let sumSin = 0;
    for (let i = 0; i < N; i++) {
      sumCos += Math.cos(theta[i]);
      sumSin += Math.sin(theta[i]);
    }
    orderParams.push(Math.sqrt(sumCos * sumCos + sumSin * sumSin) / N);

    // Update headings
    const newTheta = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let avgSin = 0;
      let avgCos = 0;
      let count = 0;
      for (let j = 0; j < N; j++) {
        let dx = x[j] - x[i];
        let dy = y[j] - y[i];
        // Periodic BC
        if (dx > BOX_SIZE / 2) dx -= BOX_SIZE;
        if (dx < -BOX_SIZE / 2) dx += BOX_SIZE;
        if (dy > BOX_SIZE / 2) dy -= BOX_SIZE;
        if (dy < -BOX_SIZE / 2) dy += BOX_SIZE;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < RADIUS) {
          avgCos += Math.cos(theta[j]);
          avgSin += Math.sin(theta[j]);
          count++;
        }
      }
      if (count > 0) {
        const avgAngle = Math.atan2(avgSin / count, avgCos / count);
        newTheta[i] = avgAngle + eta * (Math.random() - 0.5);
      } else {
        newTheta[i] = theta[i] + eta * (Math.random() - 0.5);
      }
    }

    // Update positions
    for (let i = 0; i < N; i++) {
      theta[i] = newTheta[i];
      x[i] = ((x[i] + v0 * Math.cos(theta[i])) % BOX_SIZE + BOX_SIZE) % BOX_SIZE;
      y[i] = ((y[i] + v0 * Math.sin(theta[i])) % BOX_SIZE + BOX_SIZE) % BOX_SIZE;
    }
  }

  // Return final state and order parameter history
  return {
    x: Array.from(x),
    y: Array.from(y),
    theta: Array.from(theta),
    orderParams,
  };
}

function angleToColor(angle: number): string {
  // Map angle [0, 2pi] to hue [0, 360]
  const hue = ((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const h = (hue / (2 * Math.PI)) * 360;
  return `hsl(${h}, 70%, 50%)`;
}

export function VicsekFlocking() {
  const [eta, setEta] = useState(0.5);
  const [N, setN] = useState(200);
  const [v0, setV0] = useState(0.5);
  const [seed, setSeed] = useState(0);

  const result = useMemo(() => {
    void seed;
    return runVicsek(N, eta, v0);
  }, [N, eta, v0, seed]);

  // Compute order parameter vs noise curve
  const orderVsNoise = useMemo(() => {
    void seed;
    const etaVals: number[] = [];
    const phiVals: number[] = [];
    const nSamples = 30;
    for (let i = 0; i <= nSamples; i++) {
      const etaVal = (2 * Math.PI * i) / nSamples;
      etaVals.push(etaVal);
      const res = runVicsek(N, etaVal, v0);
      const last50 = res.orderParams.slice(-50);
      phiVals.push(last50.reduce((a, b) => a + b, 0) / last50.length);
    }
    return { eta: etaVals, phi: phiVals };
  }, [N, v0, seed]);

  const handleRerun = useCallback(() => setSeed((s) => s + 1), []);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">
            Noise {'\u03B7'}: {eta.toFixed(2)}
          </label>
          <Slider
            min={0}
            max={6.28}
            step={0.1}
            value={[eta]}
            onValueChange={([v]) => setEta(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">
            Particles N: {N}
          </label>
          <Slider
            min={50}
            max={500}
            step={25}
            value={[N]}
            onValueChange={([v]) => setN(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">
            Speed v{'\u2080'}: {v0.toFixed(2)}
          </label>
          <Slider
            min={0.1}
            max={2}
            step={0.1}
            value={[v0]}
            onValueChange={([v]) => setV0(v)}
            className="w-full"
          />
        </div>
      </div>

      <button
        onClick={handleRerun}
        className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm"
      >
        Re-run
      </button>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CanvasChart
          data={[{
            x: result.x,
            y: result.y,
            type: 'scatter' as const,
            mode: 'markers' as const,
            marker: {
              size: 4,
              color: result.theta.map(angleToColor),
            },
            name: 'Particles',
          }]}
          layout={{
            title: { text: 'Particle Positions (color = heading)', font: { size: 13 } },
            xaxis: { title: { text: 'x' }, range: [0, BOX_SIZE] },
            yaxis: { title: { text: 'y' }, range: [0, BOX_SIZE], scaleanchor: 'x' },
            showlegend: false,
            margin: { t: 35, r: 15, b: 45, l: 50 },
          }}
          style={{ width: '100%', height: 350 }}
        />
        <CanvasChart
          data={[
            {
              x: orderVsNoise.eta,
              y: orderVsNoise.phi,
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              line: { color: '#8b5cf6', width: 2 },
              marker: { size: 4 },
              name: '\u03C6(\u03B7)',
            },
            {
              x: [eta, eta],
              y: [0, 1],
              type: 'scatter' as const,
              mode: 'lines' as const,
              line: { color: '#ef4444', width: 1.5 },
              name: `\u03B7 = ${eta.toFixed(1)}`,
            },
          ]}
          layout={{
            title: { text: 'Order Parameter \u03C6 vs Noise \u03B7', font: { size: 13 } },
            xaxis: { title: { text: '\u03B7' } },
            yaxis: { title: { text: '\u03C6' }, range: [0, 1.05] },
            showlegend: true,
            margin: { t: 35, r: 15, b: 45, l: 50 },
          }}
          style={{ width: '100%', height: 350 }}
        />
      </div>
    </div>
  );
}
