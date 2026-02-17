'use client';

import React, { useState, useMemo } from 'react';
import * as math from 'mathjs';
import Plotly from 'react-plotly.js';
import AppliedStatsSim2 from './applied-statistics/AppliedStatsSim2';
import AppliedStatsSim1 from './applied-statistics/AppliedStatsSim1';
import AppliedStatsSim5 from './applied-statistics/AppliedStatsSim5';
import AppliedStatsSim6 from './applied-statistics/AppliedStatsSim6';
import AppliedStatsSim7 from './applied-statistics/AppliedStatsSim7';
import AppliedStatsSim8 from './applied-statistics/AppliedStatsSim8';

interface SimulationProps {
  id: string;
}

// Central Limit Theorem Simulation
function CentralLimitTheoremSim() {
  const [sampleSize, setSampleSize] = useState(10);
  const [numSamples, setNumSamples] = useState(500);
  const data = useMemo<number[]>(() => {
    const samples = [];
    for (let i = 0; i < numSamples; i++) {
      const sample = [];
      for (let j = 0; j < sampleSize; j++) {
        sample.push(math.random(0, 1)); // uniform [0,1]
      }
      const mean = math.mean(sample);
      samples.push(mean);
    }
    return samples;
  }, [sampleSize, numSamples]);

  const histData = data.length > 0 ? data : [];
  const hist = histData.length > 0 ? {
    x: histData,
    type: 'histogram',
    nbinsx: 50,
    name: 'Sample Means'
  } : null;

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Central Limit Theorem</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-white">Sample Size: {sampleSize}</label>
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
          <label className="text-white">Number of Samples: {numSamples}</label>
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
        data={hist ? [hist as any] : []}
        layout={{
          title: { text: `Distribution of Sample Means (n=${sampleSize})` },
          xaxis: { title: { text: 'Sample Mean' } },
          yaxis: { title: { text: 'Frequency' } },
          height: 400,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' }
        }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}

// Hypothesis Testing Simulation
function HypothesisTestingSim() {
  const [group1, setGroup1] = useState('1,2,3,4,5');
  const [group2, setGroup2] = useState('2,3,4,5,6');
  const { tStat, meanDiff } = useMemo(() => {
    try {
      const g1 = group1.split(',').map(Number);
      const g2 = group2.split(',').map(Number);
      if (g1.some(isNaN) || g2.some(isNaN)) return { tStat: 0, meanDiff: 0 };

      const mean1 = math.mean(g1);
      const mean2 = math.mean(g2);
      const var1 = math.number(math.variance(g1));
      const var2 = math.number(math.variance(g2));
      const n1 = g1.length;
      const n2 = g2.length;

      const pooledVar = ((n1 - 1) * Number(var1) + (n2 - 1) * Number(var2)) / (n1 + n2 - 2);
      const se = Math.sqrt(pooledVar * (1/n1 + 1/n2));
      const t = (mean1 - mean2) / se;

      return { tStat: t, meanDiff: mean1 - mean2 };
    } catch (e) {
      console.error(e);
      return { tStat: 0, meanDiff: 0 };
    }
  }, [group1, group2]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">Hypothesis Testing (Two-Sample t-Test)</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-white">Group 1 (comma-separated):</label>
          <input
            type="text"
            value={group1}
            onChange={(e) => setGroup1(e.target.value)}
            className="w-full p-2 bg-[#0a0a15] border border-gray-600 rounded text-white"
          />
        </div>
        <div>
          <label className="text-white">Group 2 (comma-separated):</label>
          <input
            type="text"
            value={group2}
            onChange={(e) => setGroup2(e.target.value)}
            className="w-full p-2 bg-[#0a0a15] border border-gray-600 rounded text-white"
          />
        </div>
      </div>
      <div className="mb-4 text-gray-300">
        <p>t-statistic: {tStat.toFixed(3)}</p>
        <p>Mean difference: {meanDiff.toFixed(3)}</p>
        <p>Note: For significance, |t| {'>'} 2 suggests difference (approx.)</p>
      </div>
    </div>
  );
}

export const APPLIED_STATS_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'applied-stats-sim-1': AppliedStatsSim1,
  'applied-stats-sim-2': AppliedStatsSim2,
  'applied-stats-sim-3': CentralLimitTheoremSim,
  'applied-stats-sim-4': HypothesisTestingSim,
  'applied-stats-sim-5': AppliedStatsSim5,
  'applied-stats-sim-6': AppliedStatsSim6,
  'applied-stats-sim-7': AppliedStatsSim7,
  'applied-stats-sim-8': AppliedStatsSim8,
};
