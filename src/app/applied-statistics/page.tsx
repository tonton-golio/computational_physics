'use client';

import React, { useState, useEffect } from 'react';
import * as math from 'mathjs';
import Plotly from 'react-plotly.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

function LinearRegressionDemo() {
  const [noise, setNoise] = useState(1);
  const [sampleSize, setSampleSize] = useState(20);
  const [data, setData] = useState<{x: number[], y: number[], yHat: number[], residuals: number[], ciUpper: number[], ciLower: number[]}>({
    x: [],
    y: [],
    yHat: [],
    residuals: [],
    ciUpper: [],
    ciLower: []
  });

  useEffect(() => {
    const beta0 = 2;
    const beta1 = 1.5;
    const x = Array.from({length: sampleSize}, () => math.random(0, 10));
    const y = x.map(xi => beta0 + beta1 * xi + (Math.random() - 0.5) * noise * 2);
    const meanX = math.number(math.mean(x));
    const meanY = math.number(math.mean(y));
    const covXY = math.number(math.mean(x.map((xi, i) => (xi - meanX) * (y[i] - meanY))));
    const varX = math.number(math.variance(x));
    const b1 = (covXY as number) / (varX as number);
    const b0 = meanY - b1 * meanX;
    const yHat = x.map(xi => b0 + b1 * xi);
    const residuals = y.map((yi, i) => yi - yHat[i]);
    const ssRes = math.number(math.sum(residuals.map(e => e * e)));
    const sigmaHat = Math.sqrt(ssRes / (sampleSize - 2));
    const sumSqX = math.number(math.sum(x.map(xi => (xi - meanX) ** 2)));
    const t = 2; // approx for large n
    const ciUpper = x.map((xi, i) => {
      const se = sigmaHat * Math.sqrt(1 / sampleSize + (xi - meanX) ** 2 / sumSqX);
      return yHat[i] + t * se;
    });
    const ciLower = x.map((xi, i) => {
      const se = sigmaHat * Math.sqrt(1 / sampleSize + (xi - meanX) ** 2 / sumSqX);
      return yHat[i] - t * se;
    });

    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData({ x, y, yHat, residuals, ciUpper, ciLower });
  }, [noise, sampleSize]);

  const residualTraces = data.x.map((xi, i) => ({
    x: [xi, xi],
    y: [data.y[i], data.yHat[i]],
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { color: 'red', width: 1 },
    showlegend: i === 0,
    name: 'Residuals'
  }));

  const plotData = [
    {
      x: data.x,
      y: data.y,
      type: 'scatter' as const,
      mode: 'markers' as const,
      name: 'Data Points',
      marker: { color: 'blue' }
    },
    {
      x: data.x,
      y: data.yHat,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Fitted Line',
      line: { color: 'green' }
    },
    {
      x: data.x,
      y: data.ciUpper,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: 'gray' },
      showlegend: false
    },
    {
      x: data.x,
      y: data.ciLower,
      type: 'scatter' as const,
      mode: 'lines' as const,
      fill: 'tonexty',
      fillcolor: 'rgba(128,128,128,0.2)',
      line: { color: 'gray' },
      name: '95% CI'
    },
    ...residualTraces
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Linear Regression Demo</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <p className="text-sm text-gray-600">
            This demo generates random data from a linear model y = 2 + 1.5x + ε, where ε ~ N(0, σ²).
            Adjust noise (σ) and sample size to see how they affect the fit, residuals, and confidence intervals.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label>Data Noise (σ): {noise.toFixed(1)}</label>
            <input
              type="range"
              min={0.1}
              max={5}
              step={0.1}
              value={noise}
              onChange={(e) => setNoise(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label>Sample Size: {sampleSize}</label>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={sampleSize}
              onChange={(e) => setSampleSize(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
        <Plotly
          data={plotData}
          layout={{
            title: { text: 'Linear Regression: Data, Fit, Residuals, and CI' },
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'y' } },
            height: 500
          }}
          config={{ displayModeBar: false }}
        />
      </CardContent>
    </Card>
  );
}

export default function AppliedStatisticsPage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <h1 className="text-3xl font-bold text-center mb-8">Applied Statistics</h1>

      <LinearRegressionDemo />
      <NormalDistributionDemo />
      <CentralLimitTheoremDemo />
      <HypothesisTestingDemo />
    </div>
  );
}

function NormalDistributionDemo() {
  const [mean, setMean] = useState(0);
  const [sd, setSd] = useState(1);

  const x = math.range(-5, 5, 0.1).toArray();
  const y = x.map(xi => (1 / (sd * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((xi - mean) / sd) ** 2));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Normal Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label>Mean: {mean.toFixed(1)}</label>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={mean}
              onChange={(e) => setMean(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label>SD: {sd.toFixed(1)}</label>
            <input
              type="range"
              min={0.1}
              max={3}
              step={0.1}
              value={sd}
              onChange={(e) => setSd(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
        <Plotly
          data={[{
            x,
            y,
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: `N(${mean}, ${sd}²)`
          }]}
          layout={{
            title: { text: 'Normal Distribution' },
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'Density' } },
            height: 400
          }}
          config={{ displayModeBar: false }}
        />
      </CardContent>
    </Card>
  );
}

function CentralLimitTheoremDemo() {
  const [sampleSize, setSampleSize] = useState(10);
  const [numSamples, setNumSamples] = useState(500);
  const [data, setData] = useState<number[]>([]);

  useEffect(() => {
    const samples = [];
    for (let i = 0; i < numSamples; i++) {
      const sample = [];
      for (let j = 0; j < sampleSize; j++) {
        sample.push(math.random(0, 1)); // uniform [0,1]
      }
      const mean = math.mean(sample);
      samples.push(mean);
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(samples);
  }, [sampleSize, numSamples]);

  const histData = data.length > 0 ? data : [];
  const hist = histData.length > 0 ? {
    x: histData,
    type: 'histogram',
    nbinsx: 50,
    name: 'Sample Means'
  } : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Central Limit Theorem</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label>Sample Size: {sampleSize}</label>
            <input
              type="range"
              min={1}
              max={100}
              step={1}
              value={sampleSize}
              onChange={(e) => setSampleSize(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label>Number of Samples: {numSamples}</label>
            <input
              type="range"
              min={100}
              max={1000}
              step={50}
              value={numSamples}
              onChange={(e) => setNumSamples(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
        <Plotly
          data={hist ? [hist] : []}
          layout={{
            title: { text: `Distribution of Sample Means (n=${sampleSize})` },
            xaxis: { title: { text: 'Sample Mean' } },
            yaxis: { title: { text: 'Frequency' } },
            height: 400
          }}
          config={{ displayModeBar: false }}
        />
      </CardContent>
    </Card>
  );
}

function HypothesisTestingDemo() {
  const [group1, setGroup1] = useState('1,2,3,4,5');
  const [group2, setGroup2] = useState('2,3,4,5,6');
  const [tStat, setTStat] = useState(0);
  const [meanDiff, setMeanDiff] = useState(0);

  useEffect(() => {
    try {
      const g1 = group1.split(',').map(Number);
      const g2 = group2.split(',').map(Number);
      if (g1.some(isNaN) || g2.some(isNaN)) return;

      const mean1 = math.mean(g1);
      const mean2 = math.mean(g2);
      const var1 = math.number(math.variance(g1));
      const var2 = math.number(math.variance(g2));
      const n1 = g1.length;
      const n2 = g2.length;

      const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
      const se = Math.sqrt(pooledVar * (1/n1 + 1/n2));
      const t = (mean1 - mean2) / se;

      // eslint-disable-next-line react-hooks/set-state-in-effect
      setTStat(t);
      setMeanDiff(mean1 - mean2);
    } catch (e) {
      console.error(e);
    }
  }, [group1, group2]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Hypothesis Testing (Two-Sample t-Test)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label>Group 1 (comma-separated):</label>
            <input
              type="text"
              value={group1}
              onChange={(e) => setGroup1(e.target.value)}
              className="w-full p-2 border rounded"
            />
          </div>
          <div>
            <label>Group 2 (comma-separated):</label>
            <input
              type="text"
              value={group2}
              onChange={(e) => setGroup2(e.target.value)}
              className="w-full p-2 border rounded"
            />
          </div>
        </div>
        <div className="mb-4">
          <p>t-statistic: {tStat.toFixed(3)}</p>
          <p>Mean difference: {meanDiff.toFixed(3)}</p>
          <p>Note: For significance, |t| &gt; 2 suggests difference (approx.)</p>
        </div>
      </CardContent>
    </Card>
  );
}