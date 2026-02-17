'use client';

import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps { id: string }

export default function GridworldMDP({ id }: SimulationProps) { // eslint-disable-line @typescript-eslint/no-unused-vars
  const n = 6;
  const values = useMemo(() => {
    const goal = [5, 5];
    const grid: number[][] = [];
    for (let r = 0; r < n; r++) {
      const row: number[] = [];
      for (let c = 0; c < n; c++) {
        const d = Math.abs(goal[0] - r) + Math.abs(goal[1] - c);
        row.push(Math.exp(-0.45 * d));
      }
      grid.push(row);
    }
    return grid;
  }, []);

  const pathX = [0, 1, 1, 2, 3, 4, 5, 5, 5];
  const pathY = [0, 0, 1, 1, 2, 3, 3, 4, 5];

  return (
    <div className="w-full rounded-lg bg-[#151525] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-3 text-white">Gridworld MDP: Value Heatmap and Rollout</h3>
      <Plot
        data={[
          { z: values, type: 'heatmap', colorscale: 'Viridis', showscale: true },
          { x: pathX, y: pathY, type: 'scatter', mode: 'lines+markers', name: 'Greedy rollout', line: { color: '#f87171', width: 3 }, marker: { size: 6 } },
        ]}
        layout={{
          title: { text: 'Higher value near rewarding terminal state' },
          xaxis: { title: { text: 'column' }, color: '#9ca3af', range: [-0.5, 5.5] },
          yaxis: { title: { text: 'row' }, color: '#9ca3af', autorange: 'reversed', range: [-0.5, 5.5] },
          height: 430,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(15,15,25,1)',
          font: { color: '#9ca3af' },
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  );
}
