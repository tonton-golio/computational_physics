"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationToggle } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { CanvasChart, type ChartTrace, type ChartLayout } from '@/components/ui/canvas-chart';
import type { SimulationComponentProps } from '@/shared/types/simulation';

type StateType = 'vacuum' | 'coherent' | 'squeezed' | 'number' | 'cat';
type DisplayMode = 'wigner' | 'q';

// Factorial with memoization
const factCache: number[] = [1, 1];
function factorial(n: number): number {
  if (n < 0) return 1;
  if (factCache[n] !== undefined) return factCache[n];
  let r = factCache[factCache.length - 1];
  for (let i = factCache.length; i <= n; i++) {
    r *= i;
    factCache[i] = r;
  }
  return r;
}

// Laguerre polynomial L_n(x) via recurrence
function laguerre(n: number, x: number): number {
  if (n === 0) return 1;
  if (n === 1) return 1 - x;
  let l0 = 1, l1 = 1 - x;
  for (let k = 2; k <= n; k++) {
    const l2 = ((2 * k - 1 - x) * l1 - (k - 1) * l0) / k;
    l0 = l1;
    l1 = l2;
  }
  return l1;
}

// Compute Wigner function for different states
function computeWigner(
  state: StateType,
  alpha_re: number, alpha_im: number,
  squeeze_r: number, squeeze_theta: number,
  n_state: number,
  N: number, range: number,
): { W: number[][]; xvec: number[]; pvec: number[] } {
  const xvec = Array.from({ length: N }, (_, i) => -range + (2 * range * i) / (N - 1));
  const pvec = Array.from({ length: N }, (_, i) => -range + (2 * range * i) / (N - 1));

  const W: number[][] = [];

  for (let j = 0; j < N; j++) {
    const row: number[] = [];
    for (let i = 0; i < N; i++) {
      const x = xvec[i];
      const p = pvec[j];
      let val = 0;

      switch (state) {
        case 'vacuum':
          val = (1 / Math.PI) * Math.exp(-x * x - p * p);
          break;

        case 'coherent': {
          const x0 = Math.sqrt(2) * alpha_re;
          const p0 = Math.sqrt(2) * alpha_im;
          val = (1 / Math.PI) * Math.exp(-((x - x0) ** 2) - ((p - p0) ** 2));
          break;
        }

        case 'squeezed': {
          // Squeezed vacuum: elliptical Gaussian
          // Rotate coordinates by squeeze_theta/2
          const xr = x * Math.cos(squeeze_theta / 2) + p * Math.sin(squeeze_theta / 2);
          const pr = -x * Math.sin(squeeze_theta / 2) + p * Math.cos(squeeze_theta / 2);
          const sx = Math.exp(squeeze_r);
          const sp = Math.exp(-squeeze_r);
          val = (1 / Math.PI) * Math.exp(-xr * xr * sx * sx - pr * pr * sp * sp);
          break;
        }

        case 'number': {
          // W_n(x,p) = ((-1)^n / pi) * L_n(4(x^2 + p^2)) * exp(-2(x^2 + p^2))
          const r2 = x * x + p * p;
          val = (Math.pow(-1, n_state) / Math.PI) * laguerre(n_state, 4 * r2) * Math.exp(-2 * r2);
          break;
        }

        case 'cat': {
          // Cat state: N(|alpha> + |-alpha>), alpha real
          const a = alpha_re; // Use only real alpha for cat
          const x0 = Math.sqrt(2) * a;
          // W = (1/pi) * [exp(-(x-x0)^2 - p^2) + exp(-(x+x0)^2 - p^2)
          //   + 2*cos(2*p*x0*sqrt(2)) * exp(-x^2 - p^2)]  / normalization
          const norm = 2 * (1 + Math.exp(-2 * a * a)); // <cat|cat>
          const g1 = Math.exp(-((x - x0) ** 2) - p * p);
          const g2 = Math.exp(-((x + x0) ** 2) - p * p);
          const interference = 2 * Math.cos(2 * p * x0) * Math.exp(-x * x - p * p);
          val = (1 / Math.PI) * (g1 + g2 + interference) / norm * 2;
          break;
        }
      }
      row.push(val);
    }
    W.push(row);
  }
  return { W, xvec, pvec };
}

// Compute Q function: Q(alpha) = <alpha|rho|alpha>/pi
function computeQ(
  state: StateType,
  alpha_re: number, alpha_im: number,
  squeeze_r: number, squeeze_theta: number,
  n_state: number,
  N: number, range: number,
): { W: number[][]; xvec: number[]; pvec: number[] } {
  const xvec = Array.from({ length: N }, (_, i) => -range + (2 * range * i) / (N - 1));
  const pvec = Array.from({ length: N }, (_, i) => -range + (2 * range * i) / (N - 1));

  const W: number[][] = [];

  for (let j = 0; j < N; j++) {
    const row: number[] = [];
    for (let i = 0; i < N; i++) {
      // beta = (xvec[i] + i*pvec[j]) / sqrt(2)
      const beta_re = xvec[i] / Math.sqrt(2);
      const beta_im = pvec[j] / Math.sqrt(2);
      let val = 0;

      switch (state) {
        case 'vacuum':
          // Q = (1/pi) * exp(-|beta|^2)
          val = (1 / Math.PI) * Math.exp(-(beta_re ** 2 + beta_im ** 2));
          break;

        case 'coherent': {
          // Q = (1/pi) * exp(-|beta - alpha|^2)
          const dx = beta_re - alpha_re;
          const dy = beta_im - alpha_im;
          val = (1 / Math.PI) * Math.exp(-(dx * dx + dy * dy));
          break;
        }

        case 'squeezed': {
          // Q for squeezed vacuum: broader Gaussian
          const cosH = Math.cosh(squeeze_r);
          const br = beta_re * Math.cos(squeeze_theta / 2) + beta_im * Math.sin(squeeze_theta / 2);
          const bi = -beta_re * Math.sin(squeeze_theta / 2) + beta_im * Math.cos(squeeze_theta / 2);
          val = (1 / (Math.PI * cosH)) * Math.exp(-br * br / (0.5 * (1 + Math.exp(-2 * squeeze_r))) - bi * bi / (0.5 * (1 + Math.exp(2 * squeeze_r))));
          break;
        }

        case 'number': {
          // Q_n = (1/pi) * |<beta|n>|^2 = (1/pi) * (|beta|^2n / n!) * exp(-|beta|^2)
          const r2 = beta_re ** 2 + beta_im ** 2;
          val = (1 / Math.PI) * Math.pow(r2, n_state) / factorial(n_state) * Math.exp(-r2);
          break;
        }

        case 'cat': {
          // Q for cat state (|alpha> + |-alpha>)/sqrt(N)
          const a = alpha_re;
          const norm = 2 * (1 + Math.exp(-2 * a * a));
          // |<beta|alpha>|^2 + |<beta|-alpha>|^2 + 2*Re(<beta|alpha><-alpha|beta>)
          const dbp = (beta_re - a) ** 2 + beta_im ** 2;
          const dbm = (beta_re + a) ** 2 + beta_im ** 2;
          const g1 = Math.exp(-dbp);
          const g2 = Math.exp(-dbm);
          // cross term
          const cross = 2 * Math.exp(-(beta_re ** 2 + beta_im ** 2 + a * a)) * Math.cos(2 * a * beta_im);
          val = (1 / Math.PI) * (g1 + g2 + cross) / norm * 2;
          break;
        }
      }
      row.push(val);
    }
    W.push(row);
  }
  return { W, xvec, pvec };
}

// Compute photon number distribution
function computePhotonDist(
  state: StateType,
  alpha_re: number, alpha_im: number,
  squeeze_r: number,
  n_state: number,
  maxN: number,
): { ns: number[]; probs: number[] } {
  const ns = Array.from({ length: maxN + 1 }, (_, i) => i);
  const probs: number[] = [];

  const absAlpha2 = alpha_re ** 2 + alpha_im ** 2;

  for (const n of ns) {
    let p = 0;
    switch (state) {
      case 'vacuum':
        p = n === 0 ? 1 : 0;
        break;

      case 'coherent':
        p = Math.exp(-absAlpha2) * Math.pow(absAlpha2, n) / factorial(n);
        break;

      case 'squeezed':
        // Only even photon numbers
        if (n % 2 !== 0) { p = 0; break; }
        {
          const m = n / 2;
          const tanhR = Math.tanh(squeeze_r);
          p = (factorial(2 * m) / (Math.pow(4, m) * factorial(m) ** 2))
            * Math.pow(tanhR, 2 * m) / Math.cosh(squeeze_r);
        }
        break;

      case 'number':
        p = n === n_state ? 1 : 0;
        break;

      case 'cat': {
        const a = alpha_re;
        const a2 = a * a;
        const norm = 2 * (1 + Math.exp(-2 * a2));
        // |<n|cat>|^2
        // cat = (|a> + |-a>) / sqrt(norm)
        // <n|cat> = (1/sqrt(norm)) * e^{-a^2/2} * a^n / sqrt(n!) * (1 + (-1)^n)
        const factor = (1 + Math.pow(-1, n));
        if (factor === 0) { p = 0; break; }
        p = (factor * factor / norm) * Math.exp(-a2) * Math.pow(a2, n) / factorial(n);
        break;
      }
    }
    probs.push(p);
  }
  return { ns, probs };
}

const STATE_OPTIONS: { label: string; value: StateType }[] = [
  { label: 'Vacuum', value: 'vacuum' },
  { label: 'Coherent', value: 'coherent' },
  { label: 'Squeezed', value: 'squeezed' },
  { label: 'Number', value: 'number' },
  { label: 'Cat', value: 'cat' },
];

const DISPLAY_OPTIONS = [
  { label: 'Wigner', value: 'wigner' as const },
  { label: 'Q function', value: 'q' as const },
];

export default function PhaseSpaceExplorer({}: SimulationComponentProps) {
  const [stateType, setStateType] = useState<StateType>('coherent');
  const [displayMode, setDisplayMode] = useState<DisplayMode>('wigner');
  const [alphaRe, setAlphaRe] = useState(2.0);
  const [alphaIm, setAlphaIm] = useState(0.0);
  const [squeezeR, setSqueezeR] = useState(0.8);
  const [squeezeTheta, setSqueezeTheta] = useState(0);
  const [nState, setNState] = useState(1);

  const N = 100;
  const range = 5;
  const maxPhoton = 20;

  const { W, xvec, pvec } = useMemo(() => {
    const compute = displayMode === 'wigner' ? computeWigner : computeQ;
    return compute(stateType, alphaRe, alphaIm, squeezeR, squeezeTheta, nState, N, range);
  }, [stateType, displayMode, alphaRe, alphaIm, squeezeR, squeezeTheta, nState]);

  const { ns, probs } = useMemo(() => {
    return computePhotonDist(stateType, alphaRe, alphaIm, squeezeR, nState, maxPhoton);
  }, [stateType, alphaRe, alphaIm, squeezeR, nState]);

  const showAlpha = stateType === 'coherent' || stateType === 'cat';
  const showSqueeze = stateType === 'squeezed';
  const showN = stateType === 'number';

  const barTrace: ChartTrace = {
    x: ns,
    y: probs,
    type: 'bar',
    marker: { color: '#3b82f6' },
    name: 'P(n)',
  };

  const barLayout: ChartLayout = {
    title: { text: 'Photon Number Distribution' },
    xaxis: { title: { text: 'n' } },
    yaxis: { title: { text: 'P(n)' }, range: [0, Math.max(0.5, ...probs) * 1.1] },
    height: 260,
    margin: { t: 35, r: 15, b: 40, l: 45 },
  };

  return (
    <SimulationPanel title="Phase-Space Explorer" caption="Select a quantum state and visualize its Wigner or Q function alongside its photon number distribution.">
      <SimulationSettings>
        {/* State selector */}
        <div>
          <SimulationToggle
            options={STATE_OPTIONS}
            value={stateType}
            onChange={(v) => setStateType(v as StateType)}
          />
        </div>

        {/* Display mode */}
        <div>
          <SimulationToggle
            options={DISPLAY_OPTIONS}
            value={displayMode}
            onChange={(v) => setDisplayMode(v as DisplayMode)}
          />
        </div>
      </SimulationSettings>

      <SimulationConfig>
        {/* Sliders */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {showAlpha && (
            <>
              <div>
                <SimulationLabel>Re(alpha): {alphaRe.toFixed(2)}</SimulationLabel>
                <Slider value={[alphaRe]} onValueChange={(v) => setAlphaRe(v[0])} min={-4} max={4} step={0.1} className="w-full" />
              </div>
              <div>
                <SimulationLabel>Im(alpha): {alphaIm.toFixed(2)}</SimulationLabel>
                <Slider value={[alphaIm]} onValueChange={(v) => setAlphaIm(v[0])} min={-4} max={4} step={0.1} className="w-full" />
              </div>
            </>
          )}
          {showSqueeze && (
            <>
              <div>
                <SimulationLabel>Squeeze r: {squeezeR.toFixed(2)}</SimulationLabel>
                <Slider value={[squeezeR]} onValueChange={(v) => setSqueezeR(v[0])} min={0} max={2} step={0.05} className="w-full" />
              </div>
              <div>
                <SimulationLabel>Squeeze theta: {(squeezeTheta * 180 / Math.PI).toFixed(0)}°</SimulationLabel>
                <Slider value={[squeezeTheta]} onValueChange={(v) => setSqueezeTheta(v[0])} min={0} max={2 * Math.PI} step={0.05} className="w-full" />
              </div>
            </>
          )}
          {showN && (
            <div>
              <SimulationLabel>Photon number n: {nState}</SimulationLabel>
              <Slider value={[nState]} onValueChange={(v) => setNState(Math.round(v[0]))} min={0} max={8} step={1} className="w-full" />
            </div>
          )}
        </div>
      </SimulationConfig>

      <SimulationMain>
        {/* Visualizations */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasHeatmap
            data={[{
              z: W,
              x: xvec,
              y: pvec,
              type: 'heatmap',
              colorscale: 'RdBu',
              colorbar: { title: { text: displayMode === 'wigner' ? 'W(q,p)' : 'Q(q,p)' } },
            }]}
            layout={{
              title: { text: displayMode === 'wigner' ? 'Wigner Function' : 'Husimi Q Function' },
              xaxis: { title: { text: 'q' } },
              yaxis: { title: { text: 'p' }, scaleanchor: 'x' },
              height: 400,
              margin: { t: 35, r: 20, b: 45, l: 50 },
            }}
            style={{ width: '100%', height: '400px' }}
          />
          <CanvasChart
            data={[barTrace]}
            layout={barLayout}
            style={{ width: '100%', height: '400px' }}
          />
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
