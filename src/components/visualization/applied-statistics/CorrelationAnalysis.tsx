"use client";

import { useState, useMemo } from 'react';
import { mulberry32 } from '@/lib/math';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


/**
 * PDF Visualization demo.
 * Shows samples from Uniform, Poisson (approximated), Binomial, and Gaussian distributions
 * side by side, controlled by a single sample-size slider.
 */
export default function AppliedStatsSim7({}: SimulationComponentProps) {
  const [size, setSize] = useState(500);
  const [nBinomial, setNBinomial] = useState(100);
  const [pBinomial, setPBinomial] = useState(0.2);
  const [seed, setSeed] = useState(69);

  const result = useMemo(() => {
    const seededRandom = mulberry32(seed);
    const seededNormal = () => {
      const u1 = seededRandom();
      const u2 = seededRandom();
      return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    };

    // Uniform [0, 1]
    const uniform: number[] = [];
    for (let i = 0; i < size; i++) {
      uniform.push(seededRandom());
    }

    // Binomial: sum of n Bernoulli trials
    const binomial: number[] = [];
    for (let i = 0; i < size; i++) {
      let successes = 0;
      for (let j = 0; j < nBinomial; j++) {
        if (seededRandom() < pBinomial) successes++;
      }
      binomial.push(successes);
    }

    // Poisson approximation using inverse transform
    // For lambda = n*p
    const lambda = nBinomial * pBinomial;
    const poisson: number[] = [];
    for (let i = 0; i < size; i++) {
      const L = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= seededRandom();
      } while (p > L);
      poisson.push(k - 1);
    }

    // Gaussian
    const gaussian: number[] = [];
    for (let i = 0; i < size; i++) {
      gaussian.push(seededNormal());
    }

    return { uniform, binomial, poisson, gaussian, lambda };
  }, [size, nBinomial, pBinomial, seed]);

  return (
    <SimulationPanel title="Probability Density Functions" caption="Compare samples drawn from four fundamental distributions: Uniform, Poisson, Binomial, and Gaussian. Adjust the number of samples and binomial parameters to see how the histograms change.">
      <SimulationConfig>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <SimulationLabel>Samples: {size}</SimulationLabel>
            <Slider min={50} max={5000} step={50} value={[size]}
              onValueChange={([v]) => setSize(v)} />
          </div>
          <div>
            <SimulationLabel>Binomial n: {nBinomial}</SimulationLabel>
            <Slider min={5} max={200} step={5} value={[nBinomial]}
              onValueChange={([v]) => setNBinomial(v)} />
          </div>
          <div>
            <SimulationLabel>Binomial p: {pBinomial.toFixed(2)}</SimulationLabel>
            <Slider min={0.01} max={0.99} step={0.01} value={[pBinomial]}
              onValueChange={([v]) => setPBinomial(v)} />
          </div>
          <div>
            <SimulationLabel>Seed: {seed}</SimulationLabel>
            <Slider min={1} max={200} step={1} value={[seed]}
              onValueChange={([v]) => setSeed(v)} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <CanvasChart
            data={[{
              x: result.uniform,
              type: 'histogram',
              nbinsx: 40,
              marker: { color: '#f97316' }, opacity: 0.7,
              name: 'Uniform',
            }]}
            layout={{
              title: { text: 'Uniform', font: { size: 13 } },
              font: { size: 10 },
              margin: { t: 35, r: 10, b: 30, l: 35 },
              xaxis: {},
              yaxis: {},
              showlegend: false,
            }}
            style={{ width: '100%', height: 280 }}
          />
          <CanvasChart
            data={[{
              x: result.poisson,
              type: 'histogram',
              nbinsx: 40,
              marker: { color: '#ec4899' }, opacity: 0.7,
              name: 'Poisson',
            }]}
            layout={{
              title: { text: `Poisson (lambda=${result.lambda.toFixed(1)})`, font: { size: 13 } },
              font: { size: 10 },
              margin: { t: 35, r: 10, b: 30, l: 35 },
              xaxis: {},
              yaxis: {},
              showlegend: false,
            }}
            style={{ width: '100%', height: 280 }}
          />
          <CanvasChart
            data={[{
              x: result.binomial,
              type: 'histogram',
              nbinsx: 40,
              marker: { color: '#f97316' }, opacity: 0.7,
              name: 'Binomial',
            }]}
            layout={{
              title: { text: `Binomial (n=${nBinomial}, p=${pBinomial.toFixed(2)})`, font: { size: 13 } },
              font: { size: 10 },
              margin: { t: 35, r: 10, b: 30, l: 35 },
              xaxis: {},
              yaxis: {},
              showlegend: false,
            }}
            style={{ width: '100%', height: 280 }}
          />
          <CanvasChart
            data={[{
              x: result.gaussian,
              type: 'histogram',
              nbinsx: 30,
              marker: { color: '#ec4899' }, opacity: 0.7,
              name: 'Gaussian',
            }]}
            layout={{
              title: { text: 'Gaussian', font: { size: 13 } },
              font: { size: 10 },
              margin: { t: 35, r: 10, b: 30, l: 35 },
              xaxis: {},
              yaxis: {},
              showlegend: false,
            }}
            style={{ width: '100%', height: 280 }}
          />
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
