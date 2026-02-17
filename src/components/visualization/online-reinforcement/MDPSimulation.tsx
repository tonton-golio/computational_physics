'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  runMDPSimulationWorker,
  type MDPSimulationResult,
} from '@/features/simulation/simulation-worker.client';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id: string;
}

function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function runMDPSimulationFallback(
  costPerUnfilled: number,
  setupCost: number,
  maxUnfilled: number,
  alpha: number,
  timeHorizon: number
): MDPSimulationResult {
  const nProbFill = 30;
  const probFills: number[] = [];
  for (let i = 1; i <= nProbFill; i += 1) {
    probFills.push(Number((i * 0.03).toFixed(3)));
  }

  const nTrials = 20;
  const avgCosts: number[] = [];
  const minCosts: number[] = [];
  const maxCosts: number[] = [];

  for (const p of probFills) {
    let totalCostSum = 0;
    let trialMin = Infinity;
    let trialMax = -Infinity;

    for (let trial = 0; trial < nTrials; trial += 1) {
      const rng = mulberry32(trial * 1000 + Math.round(p * 10000));
      let totalCost = 0;
      let unfilled = 0;

      for (let t = 0; t < timeHorizon; t += 1) {
        if (rng() < alpha) {
          unfilled += 1;
          if (unfilled > maxUnfilled) {
            unfilled = maxUnfilled;
          }
        }

        if (unfilled > 0 && rng() < p) {
          unfilled = 0;
          totalCost += setupCost;
        } else {
          totalCost += unfilled * costPerUnfilled;
        }
      }

      totalCostSum += totalCost;
      trialMin = Math.min(trialMin, totalCost);
      trialMax = Math.max(trialMax, totalCost);
    }

    avgCosts.push(totalCostSum / nTrials);
    minCosts.push(trialMin);
    maxCosts.push(trialMax);
  }

  let optIdx = 0;
  for (let i = 1; i < avgCosts.length; i += 1) {
    if (avgCosts[i] < avgCosts[optIdx]) {
      optIdx = i;
    }
  }

  return { probFills, avgCosts, minCosts, maxCosts, optIdx };
}

export default function MDPSimulation({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [costPerUnfilled, setCostPerUnfilled] = useState(0.1);
  const [setupCost, setSetupCost] = useState(0.5);
  const [maxUnfilled, setMaxUnfilled] = useState(10);
  const [alpha, setAlpha] = useState(0.5);
  const [timeHorizon, setTimeHorizon] = useState(300);
  const [plotData, setPlotData] = useState<MDPSimulationResult | null>(null);

  useEffect(() => {
    let active = true;
    const timeout = window.setTimeout(() => {
      void runMDPSimulationWorker({
        costPerUnfilled,
        setupCost,
        maxUnfilled,
        alpha,
        timeHorizon,
      })
        .then((result) => {
          if (!active) return;
          setPlotData(result);
        })
        .catch(() => {
          if (!active) return;
          setPlotData(runMDPSimulationFallback(costPerUnfilled, setupCost, maxUnfilled, alpha, timeHorizon));
        });
    }, 80);

    return () => {
      active = false;
      window.clearTimeout(timeout);
    };
  }, [costPerUnfilled, setupCost, maxUnfilled, alpha, timeHorizon]);

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">
        MDP Example: Order Processing Cost
      </h3>
      <p className="text-gray-400 text-sm mb-4">
        Orders arrive with probability &alpha;. We can process all unfilled orders (paying setup cost K)
        or wait (paying holding cost c per unfilled order per period).
        The maximum backlog is n. Explore how the fill probability affects total cost.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
        <div>
          <label className="text-white text-xs">
            Cost/unfilled (c): {costPerUnfilled.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.01}
            max={1.0}
            step={0.01}
            value={costPerUnfilled}
            onChange={(e) => setCostPerUnfilled(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-xs">
            Setup cost (K): {setupCost.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.1}
            max={5.0}
            step={0.1}
            value={setupCost}
            onChange={(e) => setSetupCost(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-xs">
            Max unfilled (n): {maxUnfilled}
          </label>
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={maxUnfilled}
            onChange={(e) => setMaxUnfilled(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-xs">
            Order prob (&alpha;): {alpha.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.05}
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-xs">
            Horizon (T): {timeHorizon}
          </label>
          <input
            type="range"
            min={50}
            max={1000}
            step={50}
            value={timeHorizon}
            onChange={(e) => setTimeHorizon(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {!plotData ? (
        <div className="flex h-[420px] items-center justify-center rounded border border-gray-700 bg-[#0f0f19] text-sm text-gray-400">
          Running simulation...
        </div>
      ) : (
        <Plot
          data={[
            {
              x: plotData.probFills,
              y: plotData.avgCosts,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Average cost',
              line: { color: '#60a5fa', width: 2 },
              marker: { size: 5 },
            },
            {
              x: plotData.probFills.concat([...plotData.probFills].reverse()),
              y: plotData.minCosts.concat([...plotData.maxCosts].reverse()),
              type: 'scatter',
              fill: 'toself',
              fillcolor: 'rgba(96,165,250,0.15)',
              line: { color: 'transparent' },
              name: 'Min-Max range',
              showlegend: true,
            },
            {
              x: [plotData.probFills[plotData.optIdx]],
              y: [plotData.avgCosts[plotData.optIdx]],
              type: 'scatter',
              mode: 'markers',
              name: `Optimal p*=${plotData.probFills[plotData.optIdx].toFixed(3)}`,
              marker: { size: 12, color: '#4ade80', symbol: 'star' },
            },
          ]}
          layout={{
            title: { text: 'Total Cost vs Fill Probability' },
            xaxis: {
              title: { text: 'Probability of processing orders' },
              color: '#9ca3af',
            },
            yaxis: { title: { text: 'Total cost' }, color: '#9ca3af' },
            height: 420,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15,15,25,1)',
            font: { color: '#9ca3af' },
            legend: { x: 0.6, y: 1 },
            margin: { t: 40, b: 60, l: 60, r: 20 },
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      )}

      <div className="mt-3 text-xs text-gray-500">
        <p>
          Too low a fill probability accumulates holding costs; too high triggers frequent setup costs.
          The optimal policy balances these two competing costs.
        </p>
      </div>
    </div>
  );
}
