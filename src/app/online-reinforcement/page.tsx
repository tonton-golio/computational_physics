'use client';

import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

const gridSize = 4;
const actionDeltas = [
  [0, -1], // up
  [0, 1],  // down
  [-1, 0], // left
  [1, 0],  // right
];
const goal = [3, 3];
const obstacles: [number, number][] = [
  [1, 1],
  [2, 2],
];

const symbols = ['▲', '▼', '◀', '▶'];

export default function OnlineReinforcementPage() {
  const [Q, setQ] = useState<number[][][]>(
    Array(gridSize)
      .fill(0)
      .map(() =>
        Array(gridSize)
          .fill(0)
          .map(() => Array(4).fill(0))
      )
  );
  const [currentState, setCurrentState] = useState<[number, number]>([0, 0]);
  const [alpha, setAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(0.9);
  const [epsilon, setEpsilon] = useState(0.1);
  const [isRunning, setIsRunning] = useState(false);
  const [stepCount, setStepCount] = useState(0);

  const reset = () => {
    setQ(
      Array(gridSize)
        .fill(0)
        .map(() =>
          Array(gridSize)
            .fill(0)
            .map(() => Array(4).fill(0))
        )
    );
    setCurrentState([0, 0]);
    setStepCount(0);
    setIsRunning(false);
  };

  const chooseAction = (x: number, y: number): number => {
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * 4);
    } else {
      const q = Q[x][y];
      const maxQ = Math.max(...q);
      const ties = q
        .map((v, i) => (v === maxQ ? i : -1))
        .filter((i) => i !== -1);
      return ties[Math.floor(Math.random() * ties.length)];
    }
  };

  const step = () => {
    const [x, y] = currentState;
    const actionIdx = chooseAction(x, y);
    const [dx, dy] = actionDeltas[actionIdx];
    const nx = x + dx;
    const ny = y + dy;
    let reward = -1;
    let valid = true;
    if (
      nx < 0 ||
      nx >= gridSize ||
      ny < 0 ||
      ny >= gridSize ||
      obstacles.some((o) => o[0] === nx && o[1] === ny)
    ) {
      reward = -10;
      valid = false;
    } else if (nx === goal[0] && ny === goal[1]) {
      reward = 10;
    }
    // update Q
    const oldQ = Q[x][y][actionIdx];
    const maxNextQ = valid ? Math.max(...Q[nx][ny]) : 0;
    const newQ = oldQ + alpha * (reward + gamma * maxNextQ - oldQ);
    const newQTable = Q.map((row, i) =>
      row.map((stateQ, j) =>
        i === x && j === y
          ? stateQ.map((q, k) => (k === actionIdx ? newQ : q))
          : stateQ
      )
    );
    setQ(newQTable);
    if (valid) {
      setCurrentState([nx, ny]);
      if (nx === goal[0] && ny === goal[1]) {
        // reset after reaching goal
        setTimeout(() => setCurrentState([0, 0]), 1000);
      }
    }
    setStepCount((prev) => prev + 1);
  };

  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(() => {
        step();
        if (currentState[0] === goal[0] && currentState[1] === goal[1]) {
          setIsRunning(false);
        }
      }, 500);
      return () => clearInterval(interval);
    }
  }, [isRunning, currentState, Q, alpha, gamma, epsilon, step]);

  const z = Q.map((row) => row.map((stateQ) => Math.max(...stateQ)));
  const annotations: unknown[] = [];
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      let text = '';
      if (i === goal[0] && j === goal[1]) {
        text = 'G';
      } else if (obstacles.some((o) => o[0] === i && o[1] === j)) {
        text = 'X';
      } else if (i === currentState[0] && j === currentState[1]) {
        text = 'A';
      } else {
        const q = Q[i][j];
        const maxIdx = q.indexOf(Math.max(...q));
        text = symbols[maxIdx];
      }
      annotations.push({
        x: i,
        y: j,
        text: text,
        showarrow: false,
        font: { color: 'white', size: 20 },
        xref: 'x',
        yref: 'y',
      });
    }
  }

  const data = [
    {
      z: z,
      x: Array.from({ length: gridSize }, (_, i) => i),
      y: Array.from({ length: gridSize }, (_, i) => i),
      type: 'heatmap' as const,
      colorscale: 'Viridis',
      showscale: true,
    },
  ];

  const layout = {
    title: { text: 'Q-Learning Grid World' },
    xaxis: { title: { text: 'X' }, dtick: 1 },
    yaxis: { title: { text: 'Y' }, dtick: 1 },
    annotations: annotations,
    width: 600,
    height: 600,
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Online Reinforcement Learning: Q-Learning Demo</h1>
      <p className="mb-4">
        This interactive demo shows Q-learning in a 4x4 grid world. The agent (A) starts at (0,0), aims for the goal (G) at (3,3),
        avoiding obstacles (X). Arrows show the current policy (greedy action based on Q-values). Adjust parameters with sliders.
      </p>
      <div className="mb-4">
        <label>Learning Rate (α): {alpha.toFixed(2)}</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={alpha}
          onChange={(e) => setAlpha(+e.target.value)}
          className="w-full"
        />
      </div>
      <div className="mb-4">
        <label>Discount Factor (γ): {gamma.toFixed(2)}</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={gamma}
          onChange={(e) => setGamma(+e.target.value)}
          className="w-full"
        />
      </div>
      <div className="mb-4">
        <label>Exploration Epsilon: {epsilon.toFixed(2)}</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={epsilon}
          onChange={(e) => setEpsilon(+e.target.value)}
          className="w-full"
        />
      </div>
      <div className="mb-4">
        <button onClick={reset} className="mr-2 px-4 py-2 bg-blue-500 text-white rounded">
          Reset
        </button>
        <button onClick={step} disabled={isRunning} className="mr-2 px-4 py-2 bg-green-500 text-white rounded">
          Step
        </button>
        <button
          onClick={() => setIsRunning(!isRunning)}
          className="px-4 py-2 bg-red-500 text-white rounded"
        >
          {isRunning ? 'Stop' : 'Run'}
        </button>
      </div>
      <p>Step Count: {stepCount}</p>
      <Plot data={data} layout={layout} />
      <div className="mt-4">
        <h2 className="text-xl font-semibold">How it works</h2>
        <p>
          Q-learning updates Q-values using: Q(s,a) ← Q(s,a) + α [r + γ max Q(s&apos;,a&apos;) - Q(s,a)].
          The heatmap shows the maximum Q-value per state. Policy arrows indicate the best action.
        </p>
      </div>
    </div>
  );
}