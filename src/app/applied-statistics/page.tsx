import React, { useState, useEffect } from 'react';
import * as math from 'mathjs';
import Plotly from 'react-plotly.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function AppliedStatisticsPage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <h1 className="text-3xl font-bold text-center mb-8">Applied Statistics</h1>

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
            type: 'scatter',
            mode: 'lines',
            name: `N(${mean}, ${sd}²)`
          }]}
          layout={{
            title: 'Normal Distribution',
            xaxis: { title: 'x' },
            yaxis: { title: 'Density' },
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
            title: `Distribution of Sample Means (n=${sampleSize})`,
            xaxis: { title: 'Sample Mean' },
            yaxis: { title: 'Frequency' },
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
      const var1 = math.var(g1);
      const var2 = math.var(g2);
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