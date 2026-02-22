"use client";

import { useState, useEffect } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

// Box-Muller transform for generating normal random variables
function boxMuller(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export default function RegressionAnalysis({}: SimulationComponentProps) {
  const [mean, setMean] = useState(0);
  const [stdDev, setStdDev] = useState(1);
  const [numSamples, setNumSamples] = useState(1000);
  const [samples, setSamples] = useState<number[]>([]);

  // Generate samples when parameters change
  useEffect(() => {
    const newSamples = Array.from({ length: numSamples }, () => mean + stdDev * boxMuller());
    setSamples(newSamples);
  }, [mean, stdDev, numSamples]);

  // Calculate theoretical normal PDF
  const xMin = mean - 4 * stdDev;
  const xMax = mean + 4 * stdDev;
  const xValues = [];
  const pdfValues = [];
  const numPoints = 200;

  for (let i = 0; i <= numPoints; i++) {
    const x = xMin + (i / numPoints) * (xMax - xMin);
    xValues.push(x);
    const pdf = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mean) / stdDev) ** 2);
    pdfValues.push(pdf);
  }

  const plotData = [
    {
      x: samples,
      type: 'histogram' as const,
      nbinsx: 50,
      name: 'Sample Histogram',
      opacity: 0.7,
      marker: { color: '#007bff' },
      histnorm: 'probability density' as const,
    },
    {
      x: xValues,
      y: pdfValues,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: `Normal PDF (mu=${mean.toFixed(1)}, sigma=${stdDev.toFixed(1)})`,
      line: { color: 'red', width: 2 },
    },
  ];

  return (
    <SimulationPanel title="Interactive Normal Distribution Simulation" caption="Adjust the mean and standard deviation to see how the theoretical normal distribution (red line) fits the histogram of randomly generated samples.">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <SimulationLabel>Mean (mu): {mean.toFixed(1)}</SimulationLabel>
            <Slider
              min={-3}
              max={3}
              step={0.1}
              value={[mean]}
              onValueChange={([v]) => setMean(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Standard Deviation (sigma): {stdDev.toFixed(1)}</SimulationLabel>
            <Slider
              min={0.1}
              max={3}
              step={0.1}
              value={[stdDev]}
              onValueChange={([v]) => setStdDev(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Number of Samples: {numSamples}</SimulationLabel>
            <Slider
              min={100}
              max={5000}
              step={100}
              value={[numSamples]}
              onValueChange={([v]) => setNumSamples(v)}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
      <CanvasChart
        data={plotData}
        layout={{
          title: {
            text: 'Normal Distribution: Samples vs Theoretical PDF',
          },
          xaxis: {
            title: { text: 'Value' },
          },
          yaxis: {
            title: { text: 'Density' },
          },
          height: 500,
        }}
        style={{ width: '100%' }}
      />
      </SimulationMain>
    </SimulationPanel>
  );
}
