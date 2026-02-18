'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { usePlotlyTheme } from '@/lib/plotly-theme';
import { Slider } from '@/components/ui/slider';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface NetworkNode {
  id: number;
  degree: number;
  group: number;
  x: number;
  y: number;
}

interface NetworkLink {
  source: number;
  target: number;
}

function generateScaleFreeNetwork(n: number): { nodes: NetworkNode[]; links: NetworkLink[] } {
  const nodeList: NetworkNode[] = [];
  const linkList: NetworkLink[] = [];

  // Start with a small complete graph (triangle)
  for (let i = 0; i < Math.min(3, n); i++) {
    nodeList.push({
      id: i,
      degree: Math.min(3, n) - 1,
      group: 1,
      x: Math.cos(2 * Math.PI * i / Math.min(3, n)),
      y: Math.sin(2 * Math.PI * i / Math.min(3, n)),
    });
  }

  if (n >= 3) {
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) {
        linkList.push({ source: i, target: j });
      }
    }
  }

  // Add remaining nodes with preferential attachment (Barabasi-Albert)
  for (let i = 3; i < n; i++) {
    const degrees = nodeList.map(node => node.degree);
    const totalDegree = degrees.reduce((sum, d) => sum + d, 0);
    const probabilities = degrees.map(d => d / totalDegree);

    const m = 2;
    const targets = new Set<number>();

    for (let k = 0; k < m && targets.size < nodeList.length; k++) {
      const rand = Math.random();
      let cumulative = 0;
      for (let j = 0; j < nodeList.length; j++) {
        cumulative += probabilities[j];
        if (rand <= cumulative && !targets.has(j)) {
          targets.add(j);
          linkList.push({ source: i, target: j });
          nodeList[j].degree++;
          break;
        }
      }
    }

    nodeList.push({
      id: i,
      degree: targets.size,
      group: Math.floor(Math.random() * 3) + 1,
      x: (Math.random() - 0.5) * 2,
      y: (Math.random() - 0.5) * 2,
    });
  }

  return { nodes: nodeList, links: linkList };
}

// Simple force-directed layout (a few iterations of Fruchterman-Reingold)
function layoutNetwork(nodes: NetworkNode[], links: NetworkLink[], iterations: number = 50): NetworkNode[] {
  const positioned = nodes.map(n => ({ ...n }));
  const k = Math.sqrt(4 / Math.max(1, nodes.length));

  for (let iter = 0; iter < iterations; iter++) {
    const temp = 1 - iter / iterations;
    const dx = new Array(positioned.length).fill(0);
    const dy = new Array(positioned.length).fill(0);

    // Repulsion
    for (let i = 0; i < positioned.length; i++) {
      for (let j = i + 1; j < positioned.length; j++) {
        const ddx = positioned[i].x - positioned[j].x;
        const ddy = positioned[i].y - positioned[j].y;
        const dist = Math.max(0.01, Math.sqrt(ddx * ddx + ddy * ddy));
        const force = (k * k) / dist;
        dx[i] += (ddx / dist) * force;
        dy[i] += (ddy / dist) * force;
        dx[j] -= (ddx / dist) * force;
        dy[j] -= (ddy / dist) * force;
      }
    }

    // Attraction along edges
    for (const link of links) {
      const si = link.source;
      const ti = link.target;
      if (si >= positioned.length || ti >= positioned.length) continue;
      const ddx = positioned[si].x - positioned[ti].x;
      const ddy = positioned[si].y - positioned[ti].y;
      const dist = Math.max(0.01, Math.sqrt(ddx * ddx + ddy * ddy));
      const force = (dist * dist) / k;
      dx[si] -= (ddx / dist) * force;
      dy[si] -= (ddy / dist) * force;
      dx[ti] += (ddx / dist) * force;
      dy[ti] += (ddy / dist) * force;
    }

    // Apply with temperature
    for (let i = 0; i < positioned.length; i++) {
      const disp = Math.max(0.01, Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]));
      const maxDisp = temp * 0.1;
      positioned[i].x += (dx[i] / disp) * Math.min(disp, maxDisp);
      positioned[i].y += (dy[i] / disp) * Math.min(disp, maxDisp);
    }
  }

  return positioned;
}

export function ScaleFreeNetwork() {
  const [numNodes, setNumNodes] = useState(50);
  const [seed, setSeed] = useState(0);
  const { mergeLayout } = usePlotlyTheme();

  const network = useMemo(() => {
    const raw = generateScaleFreeNetwork(numNodes);
    const positioned = layoutNetwork(raw.nodes, raw.links);
    return { nodes: positioned, links: raw.links };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numNodes, seed]);

  const colors = ['#3b82f6', '#ec4899', '#10b981'];

  // Build edge traces
  const edgeX: (number | null)[] = [];
  const edgeY: (number | null)[] = [];
  for (const link of network.links) {
    const s = network.nodes[link.source];
    const t = network.nodes[link.target];
    if (s && t) {
      edgeX.push(s.x, t.x, null);
      edgeY.push(s.y, t.y, null);
    }
  }

  // Degree distribution
  const degrees = network.nodes.map(n => n.degree);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Nodes: {numNodes}</label>
          <Slider
            min={10}
            max={100}
            step={5}
            value={[numNodes]}
            onValueChange={([v]) => setNumNodes(v)}
            className="w-48"
          />
        </div>
        <button
          onClick={() => setSeed(s => s + 1)}
          className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4"
        >
          Regenerate
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Plot
          data={[
            {
              x: edgeX,
              y: edgeY,
              mode: 'lines',
              type: 'scatter',
              line: { color: '#4b5563', width: 0.8 },
              showlegend: false,
              hoverinfo: 'skip',
            },
            {
              x: network.nodes.map(n => n.x),
              y: network.nodes.map(n => n.y),
              mode: 'markers',
              type: 'scatter',
              marker: {
                size: network.nodes.map(n => Math.max(4, Math.min(18, 3 + n.degree * 1.5))),
                color: network.nodes.map(n => colors[(n.group - 1) % colors.length]),
                line: { color: '#ffffff', width: 0.5 },
              },
              text: network.nodes.map(n => `Node ${n.id}, degree: ${n.degree}`),
              showlegend: false,
            },
          ]}
          layout={mergeLayout({
            title: { text: `Scale-Free Network (n=${numNodes})`, font: { size: 13 } },
            xaxis: { showticklabels: false, showgrid: false, zeroline: false },
            yaxis: { showticklabels: false, showgrid: false, zeroline: false, scaleanchor: 'x' },
            margin: { t: 40, r: 10, b: 10, l: 10 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 400 }}
        />
        <Plot
          data={[{
            x: degrees,
            type: 'histogram',
            marker: { color: '#3b82f6' },
          }]}
          layout={mergeLayout({
            title: { text: 'Degree Distribution', font: { size: 13 } },
            xaxis: { title: { text: 'Degree' } },
            yaxis: { title: { text: 'Frequency' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
          })}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: 400 }}
        />
      </div>
    </div>
  );
}
