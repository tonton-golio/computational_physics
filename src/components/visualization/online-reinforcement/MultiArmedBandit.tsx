'use client';

import React, { useState, useMemo, useCallback } from 'react';
import dynamic from 'next/dynamic';

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

type Method = 'greedy' | 'epsilon-greedy' | 'UCB' | 'LCB';

function banditRun(
  probs: number[],
  T: number,
  method: Method,
  epsilon: number,
  rng: () => number
): number[] {
  const nArms = probs.length;
  const n = new Array(nArms).fill(0); // pull counts
  const X = new Array(nArms).fill(0); // cumulative reward per arm
  const rewards = new Array(T).fill(0); // cumulative reward over time

  // Initialize: play each arm once
  for (let arm = 0; arm < nArms && arm < T; arm++) {
    const reward = rng() < probs[arm] ? 1 : 0;
    n[arm] = 1;
    X[arm] = reward;
    rewards[arm] = arm > 0 ? rewards[arm - 1] + reward : reward;
  }

  for (let t = nArms; t < T; t++) {
    let chosenArm = 0;

    if (method === 'greedy') {
      // Pick arm with highest empirical mean
      let bestMean = -Infinity;
      for (let a = 0; a < nArms; a++) {
        const mean = n[a] > 0 ? X[a] / n[a] : 0;
        if (mean > bestMean) {
          bestMean = mean;
          chosenArm = a;
        }
      }
    } else if (method === 'epsilon-greedy') {
      if (rng() < epsilon) {
        chosenArm = Math.floor(rng() * nArms);
      } else {
        let bestMean = -Infinity;
        for (let a = 0; a < nArms; a++) {
          const mean = n[a] > 0 ? X[a] / n[a] : 0;
          if (mean > bestMean) {
            bestMean = mean;
            chosenArm = a;
          }
        }
      }
    } else if (method === 'UCB') {
      let bestScore = -Infinity;
      for (let a = 0; a < nArms; a++) {
        const mean = n[a] > 0 ? X[a] / n[a] : 0;
        const bonus = n[a] > 0 ? Math.sqrt(2 * Math.log(t + 1) / n[a]) : Infinity;
        const score = mean + bonus;
        if (score > bestScore) {
          bestScore = score;
          chosenArm = a;
        }
      }
    } else if (method === 'LCB') {
      // LCB: penalize arms that have been played more -- pick the arm with the
      // lowest lower confidence bound (used in cost-minimization framing)
      // In a reward-maximization setting we still pick the arm whose LCB is highest
      let bestScore = -Infinity;
      for (let a = 0; a < nArms; a++) {
        const mean = n[a] > 0 ? X[a] / n[a] : 0;
        const penalty =
          n[a] > 0 ? Math.sqrt((3 * Math.log(t + 1)) / (2 * n[a])) : Infinity;
        const score = mean - penalty;
        if (score > bestScore) {
          bestScore = score;
          chosenArm = a;
        }
      }
    }

    const reward = rng() < probs[chosenArm] ? 1 : 0;
    n[chosenArm] += 1;
    X[chosenArm] += reward;
    rewards[t] = rewards[t - 1] + reward;
  }

  return rewards;
}

export default function MultiArmedBandit({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const [T, setT] = useState(500);
  const [nExperiments, setNExperiments] = useState(20);
  const [epsilon, setEpsilon] = useState(0.1);
  const [seed, setSeed] = useState(42);

  const probs = useMemo(() => [0.1, 0.15, 0.2, 0.5], []);
  const methods: Method[] = useMemo(
    () => ['greedy', 'epsilon-greedy', 'UCB', 'LCB'],
    []
  );

  const rerun = useCallback(() => {
    setSeed((prev) => prev + 1);
  }, []);

  const plotData = useMemo(() => {
    const results: Record<string, number[][]> = {};
    let seedCounter = seed;

    for (const method of methods) {
      results[method] = [];
      for (let i = 0; i < nExperiments; i++) {
        const rng = mulberry32(seedCounter++);
        results[method].push(banditRun(probs, T, method, epsilon, rng));
      }
    }

    // Average across experiments
    const avgRewards: Record<string, number[]> = {};
    for (const method of methods) {
      avgRewards[method] = new Array(T).fill(0);
      for (let t = 0; t < T; t++) {
        let sum = 0;
        for (let i = 0; i < nExperiments; i++) {
          sum += results[method][i][t];
        }
        avgRewards[method][t] = sum / nExperiments;
      }
    }

    const timeSteps = Array.from({ length: T }, (_, i) => i);
    return { avgRewards, timeSteps };
  }, [T, nExperiments, epsilon, probs, methods, seed]);

  const colors: Record<string, string> = {
    greedy: '#f87171',
    'epsilon-greedy': '#60a5fa',
    UCB: '#4ade80',
    LCB: '#c084fc',
  };

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">
        Multi-Armed Bandit: Strategy Comparison
      </h3>
      <p className="text-gray-400 text-sm mb-4">
        A 4-armed bandit with Bernoulli rewards and win-probabilities [0.1, 0.15, 0.2, 0.5].
        Compare the cumulative reward of Greedy, &epsilon;-Greedy, UCB, and LCB strategies.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        <div>
          <label className="text-white text-sm">Time horizon (T): {T}</label>
          <input
            type="range"
            min={100}
            max={2000}
            step={100}
            value={T}
            onChange={(e) => setT(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-sm">
            Experiments: {nExperiments}
          </label>
          <input
            type="range"
            min={1}
            max={100}
            step={1}
            value={nExperiments}
            onChange={(e) => setNExperiments(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-white text-sm">
            Epsilon (&epsilon;): {epsilon.toFixed(2)}
          </label>
          <input
            type="range"
            min={0.01}
            max={0.5}
            step={0.01}
            value={epsilon}
            onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <div className="flex items-end">
          <button
            onClick={rerun}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm transition-colors"
          >
            Re-run
          </button>
        </div>
      </div>

      <Plot
        data={methods.map((method) => ({
          x: plotData.timeSteps,
          y: plotData.avgRewards[method],
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: method,
          line: { color: colors[method], width: 2 },
        }))}
        layout={{
          title: {
            text: `Average Cumulative Reward (${nExperiments} experiments)`,
          },
          xaxis: { title: { text: 'Time step' }, color: '#9ca3af' },
          yaxis: {
            title: { text: 'Cumulative reward' },
            color: '#9ca3af',
          },
          height: 450,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
          legend: { x: 0.02, y: 1 },
          margin: { t: 40, b: 60, l: 60, r: 20 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />

      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-400">
        <div>
          <strong className="text-white">Arm probabilities:</strong>{' '}
          {probs.map((p, i) => (
            <span key={i} className="mr-2">
              Arm {i + 1}: {p}
            </span>
          ))}
        </div>
        <div>
          <strong className="text-white">Optimal arm:</strong> Arm 4 (p=0.5),
          yielding expected cumulative reward of {(0.5 * T).toFixed(0)} over {T}{' '}
          steps
        </div>
      </div>
    </div>
  );
}
