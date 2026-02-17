'use client';

import React, { useState, useEffect } from 'react';
import Plotly from 'react-plotly.js';

// Box-Muller transform for generating normal random variables
function boxMuller(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

interface AppliedStatsSim2Props {
  id: string;
}

function AppliedStatsSim2({ id }: AppliedStatsSim2Props) {
  const [mean, setMean] = useState(0);
  const [stdDev, setStdDev] = useState(1);
  const [numSamples, setNumSamples] = useState(1000);
  const [samples, setSamples] = useState<number[]>([]);

  // Generate samples when parameters change
  useEffect(() => {
    const newSamples = Array.from({ length: numSamples }, () => mean + stdDev * boxMuller());
    // eslint-disable-next-line react-hooks/set-state-in-effect
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
      marker: { color: 'rgba(0, 123, 255, 0.7)' },
      histnorm: 'probability density' as const,
    },
    {
      x: xValues,
      y: pdfValues,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: `Normal PDF (μ=${mean.toFixed(1)}, σ=${stdDev.toFixed(1)})`,
      line: { color: 'red', width: 2 },
    },
  ];

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Interactive Normal Distribution Simulation</h3>
      <div className="mb-4">
        <p className="text-sm text-gray-300">
          Adjust the mean and standard deviation to see how the theoretical normal distribution (red line) fits the histogram of randomly generated samples.
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="text-white block mb-1">Mean (μ): {mean.toFixed(1)}</label>
          <input
            type="range"
            min={-3}
            max={3}
            step={0.1}
            value={mean}
            onChange={(e) => setMean(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white block mb-1">Standard Deviation (σ): {stdDev.toFixed(1)}</label>
          <input
            type="range"
            min={0.1}
            max={3}
            step={0.1}
            value={stdDev}
            onChange={(e) => setStdDev(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white block mb-1">Number of Samples: {numSamples}</label>
          <input
            type="range"
            min={100}
            max={5000}
            step={100}
            value={numSamples}
            onChange={(e) => setNumSamples(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </div>
      <Plotly
        data={plotData}
        layout={{
          title: {
            text: 'Normal Distribution: Samples vs Theoretical PDF',
            font: { color: '#ffffff' }
          },
          xaxis: {
            title: { text: 'Value', font: { color: '#9ca3af' } },
            tickfont: { color: '#9ca3af' },
            gridcolor: '#374151'
          },
          yaxis: {
            title: { text: 'Density', font: { color: '#9ca3af' } },
            tickfont: { color: '#9ca3af' },
            gridcolor: '#374151'
          },
          height: 500,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          legend: { font: { color: '#9ca3af' } }
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
      <div className="mt-4 text-sm text-gray-400">
        <p>The histogram shows the distribution of {numSamples} random samples drawn from a normal distribution.</p>
        <p>The red line represents the theoretical probability density function.</p>
      </div>
    </div>
  );
}

export default AppliedStatsSim2;