'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import Plotly from 'plotly.js-dist';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';

const COLORS = {
  primary: '#3b82f6',
  secondary: '#ec4899',
  tertiary: '#10b981',
  warning: '#f59e0b',
  danger: '#ef4444',
  accent: '#8b5cf6',
  grid: '#1e1e2e',
  zero: '#2d2d44',
};

const BASE_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(15,15,25,1)',
  font: { color: '#9ca3af', family: 'system-ui' },
  margin: { t: 40, r: 20, b: 40, l: 50 },
  xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.zero },
  yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.zero },
};

interface SimulationProps {
  id?: string;
}

// ============ COMPLEX PHYSICS SIMULATIONS ============

// 1. Percolation Simulation
export function PercolationSimulation({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [p, setP] = useState(0.5);
  const [size, setSize] = useState(20);
  const [grid, setGrid] = useState<number[][]>([]);

  const generateGrid = useCallback((probability: number, gridSize: number) => {
    const newGrid = [];
    for (let i = 0; i < gridSize; i++) {
      const row = [];
      for (let j = 0; j < gridSize; j++) {
        row.push(Math.random() < probability ? 1 : 0);
      }
      newGrid.push(row);
    }
    return newGrid;
  }, []);

  const findClusters = useCallback((lattice: number[][]) => {
    const n = lattice.length;
    const visited = Array.from({ length: n }, () => Array(n).fill(false));
    const clusters: number[][] = [];

    function dfs(i: number, j: number, cluster: number[][]) {
      if (i < 0 || i >= n || j < 0 || j >= n || visited[i][j] || lattice[i][j] === 0) return;
      visited[i][j] = true;
      cluster.push([i, j]);
      dfs(i + 1, j, cluster);
      dfs(i - 1, j, cluster);
      dfs(i, j + 1, cluster);
      dfs(i, j - 1, cluster);
    }

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!visited[i][j] && lattice[i][j] === 1) {
          const cluster: number[][] = [];
          dfs(i, j, cluster);
          if (cluster.length > 0) clusters.push(cluster);
        }
      }
    }

    return clusters;
  }, []);

  const updateVisualization = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const currentGrid = generateGrid(p, size);
    setGrid(currentGrid);

    const clusters = findClusters(currentGrid);
    const clusterSizes = clusters.map(c => c.length);
    const maxClusterSize = Math.max(...clusterSizes, 0);
    const percolates = clusters.some(cluster =>
      cluster.some(([i]) => i === 0) && cluster.some(([i]) => i === size - 1)
    );

    // Color by cluster
    const clusterColors = clusters.map((cluster, idx) => {
      const size = cluster.length;
      const intensity = Math.min(size / maxClusterSize, 1);
      return `rgba(${Math.floor(100 + 155 * intensity)}, ${Math.floor(100 + 155 * (1 - intensity))}, 255, 0.8)`;
    });

    const data: Plotly.Data[] = [{
      z: currentGrid,
      type: 'heatmap',
      colorscale: [
        [0, '#1a1a2e'],
        [1, percolates ? '#ef4444' : '#3b82f6']
      ],
      showscale: false,
      hoverongaps: false,
    }];

    // Add cluster boundaries
    clusters.forEach((cluster, idx) => {
      const x = cluster.map(([i, j]) => j);
      const y = cluster.map(([i, j]) => i);
      data.push({
        x,
        y,
        mode: 'markers',
        type: 'scatter',
        marker: {
          size: 8,
          color: clusterColors[idx],
          line: { width: 1, color: '#ffffff' }
        },
        showlegend: false,
        hoverinfo: 'skip',
      });
    });

    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      title: {
        text: `Percolation (p=${p.toFixed(2)}) - ${percolates ? 'Percolates!' : 'No percolation'}`,
        font: { color: percolates ? '#ef4444' : '#9ca3af' }
      },
      xaxis: { ...BASE_LAYOUT.xaxis, showticklabels: false },
      yaxis: { ...BASE_LAYOUT.yaxis, showticklabels: false, autorange: 'reversed' },
      width: 500,
      height: 500,
    };

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [p, size, generateGrid, findClusters]);

  useEffect(() => {
    updateVisualization();
  }, [updateVisualization]);

  return (
    <div className="space-y-4">
      <div className="flex gap-4 flex-wrap">
        <div className="flex-1 min-w-32">
          <label className="text-sm text-gray-300 mb-2 block">
            Probability p: {p.toFixed(2)}
          </label>
          <Slider
            value={[p]}
            onValueChange={(val) => setP(val[0])}
            min={0}
            max={1}
            step={0.01}
            className="w-full"
          />
        </div>
        <div className="flex-1 min-w-32">
          <label className="text-sm text-gray-300 mb-2 block">
            Grid Size: {size}
          </label>
          <Slider
            value={[size]}
            onValueChange={(val) => setSize(val[0])}
            min={10}
            max={50}
            step={5}
            className="w-full"
          />
        </div>
        <Button onClick={updateVisualization} className="mt-6">
          Regenerate
        </Button>
      </div>
      <div ref={containerRef} className="w-full h-96 bg-[#151525] rounded-lg overflow-hidden" />
      <p className="text-sm text-gray-400">
        Site percolation on a square lattice. Blue cells are occupied. Red indicates percolation through the system.
      </p>
    </div>
  );
}

// 2. Ising Model Simulation
export function IsingModel({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [temperature, setTemperature] = useState(2.0);
  const [size, setSize] = useState(20);
  const [spins, setSpins] = useState<number[][]>([]);
  const [running, setRunning] = useState(false);
  const [energy, setEnergy] = useState(0);
  const [magnetization, setMagnetization] = useState(0);

  const initializeSpins = useCallback((gridSize: number) => {
    const newSpins = [];
    for (let i = 0; i < gridSize; i++) {
      const row = [];
      for (let j = 0; j < gridSize; j++) {
        row.push(Math.random() > 0.5 ? 1 : -1);
      }
      newSpins.push(row);
    }
    return newSpins;
  }, []);

  const calculateEnergy = useCallback((spinGrid: number[][]) => {
    let totalEnergy = 0;
    const n = spinGrid.length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const neighbors = [
          spinGrid[(i + 1) % n][j],
          spinGrid[i][(j + 1) % n],
          spinGrid[(i - 1 + n) % n][j],
          spinGrid[i][(j - 1 + n) % n],
        ];
        totalEnergy -= spinGrid[i][j] * neighbors.reduce((sum, neighbor) => sum + neighbor, 0);
      }
    }
    return totalEnergy / 2; // Each pair counted twice
  }, []);

  const calculateMagnetization = useCallback((spinGrid: number[][]) => {
    const total = spinGrid.flat().reduce((sum, spin) => sum + spin, 0);
    return total / (spinGrid.length * spinGrid.length);
  }, []);

  const monteCarloStep = useCallback((spinGrid: number[][], T: number) => {
    const n = spinGrid.length;
    const newGrid = spinGrid.map(row => [...row]);

    // Single spin flip attempt
    const i = Math.floor(Math.random() * n);
    const j = Math.floor(Math.random() * n);

    const currentSpin = newGrid[i][j];
    const neighbors = [
      newGrid[(i + 1) % n][j],
      newGrid[i][(j + 1) % n],
      newGrid[(i - 1 + n) % n][j],
      newGrid[i][(j - 1 + n) % n],
    ];
    const deltaE = 2 * currentSpin * neighbors.reduce((sum, neighbor) => sum + neighbor, 0);

    if (deltaE <= 0 || Math.random() < Math.exp(-deltaE / T)) {
      newGrid[i][j] = -currentSpin;
    }

    return newGrid;
  }, []);

  const updateVisualization = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const data: Plotly.Data[] = [{
      z: spins,
      type: 'heatmap',
      colorscale: [
        [-1, '#1e40af'], // Blue for down spins
        [0, '#1a1a2e'],  // Black for zero
        [1, '#dc2626']   // Red for up spins
      ],
      showscale: false,
      hoverongaps: false,
    }];

    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      title: {
        text: `Ising Model (T=${temperature.toFixed(2)}, M=${magnetization.toFixed(3)}, E=${energy.toFixed(1)})`,
      },
      xaxis: { ...BASE_LAYOUT.xaxis, showticklabels: false },
      yaxis: { ...BASE_LAYOUT.yaxis, showticklabels: false, autorange: 'reversed' },
      width: 500,
      height: 500,
    };

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [spins, temperature, energy, magnetization]);

  useEffect(() => {
    if (spins.length === 0) {
      const initialSpins = initializeSpins(size);
      setSpins(initialSpins);
      setEnergy(calculateEnergy(initialSpins));
      setMagnetization(calculateMagnetization(initialSpins));
    }
  }, [size, initializeSpins, calculateEnergy, calculateMagnetization]);

  useEffect(() => {
    if (running) {
      const interval = setInterval(() => {
        setSpins(currentSpins => {
          const newSpins = monteCarloStep(currentSpins, temperature);
          setEnergy(calculateEnergy(newSpins));
          setMagnetization(calculateMagnetization(newSpins));
          return newSpins;
        });
      }, 50);
      return () => clearInterval(interval);
    }
  }, [running, temperature, monteCarloStep, calculateEnergy, calculateMagnetization]);

  useEffect(() => {
    if (spins.length > 0) {
      updateVisualization();
    }
  }, [spins, updateVisualization]);

  const reset = () => {
    setRunning(false);
    const initialSpins = initializeSpins(size);
    setSpins(initialSpins);
    setEnergy(calculateEnergy(initialSpins));
    setMagnetization(calculateMagnetization(initialSpins));
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-4 flex-wrap">
        <div className="flex-1 min-w-32">
          <label className="text-sm text-gray-300 mb-2 block">
            Temperature: {temperature.toFixed(2)}
          </label>
          <Slider
            value={[temperature]}
            onValueChange={(val) => setTemperature(val[0])}
            min={0.1}
            max={5}
            step={0.1}
            className="w-full"
          />
        </div>
        <div className="flex-1 min-w-32">
          <label className="text-sm text-gray-300 mb-2 block">
            Grid Size: {size}
          </label>
          <Slider
            value={[size]}
            onValueChange={(val) => {
              setSize(val[0]);
              reset();
            }}
            min={10}
            max={40}
            step={5}
            className="w-full"
          />
        </div>
        <div className="flex gap-2 mt-6">
          <Button onClick={() => setRunning(!running)}>
            {running ? 'Pause' : 'Start'}
          </Button>
          <Button onClick={reset} variant="outline">
            Reset
          </Button>
        </div>
      </div>
      <div ref={containerRef} className="w-full h-96 bg-[#151525] rounded-lg overflow-hidden" />
      <p className="text-sm text-gray-400">
        Ising model simulation using Metropolis Monte Carlo. Red = up spin (+1), Blue = down spin (-1).
        Magnetization M and Energy E shown in title.
      </p>
    </div>
  );
}

// 3. Network Visualization (Scale-free network generation)
export function ScaleFreeNetwork({}: SimulationProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes] = useState(50);
  const [network, setNetwork] = useState<{ nodes: any[], links: any[] }>({ nodes: [], links: [] });

  const generateScaleFreeNetwork = useCallback((n: number) => {
    const nodeList = [];
    const linkList = [];

    // Start with a small complete graph
    for (let i = 0; i < Math.min(3, n); i++) {
      nodeList.push({
        id: i,
        degree: Math.min(3, n) - 1,
        group: 1,
        x: Math.cos(2 * Math.PI * i / Math.min(3, n)),
        y: Math.sin(2 * Math.PI * i / Math.min(3, n)),
      });
    }

    // Add links for initial triangle
    if (n >= 3) {
      for (let i = 0; i < 3; i++) {
        for (let j = i + 1; j < 3; j++) {
          linkList.push({ source: i, target: j });
        }
      }
    }

    // Add remaining nodes with preferential attachment
    for (let i = 3; i < n; i++) {
      const degrees = nodeList.map(node => node.degree);
      const totalDegree = degrees.reduce((sum, d) => sum + d, 0);

      // Preferential attachment: probability proportional to degree
      const probabilities = degrees.map(d => d / totalDegree);

      // Connect to m existing nodes (m=2 for Barabasi-Albert)
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
  }, []);

  useEffect(() => {
    const net = generateScaleFreeNetwork(nodes);
    setNetwork(net);

    const container = containerRef.current;
    if (!container) return;

    const data: Plotly.Data[] = [
      // Links
      {
        x: net.links.map(link => [net.nodes[link.source].x, net.nodes[link.target].x]).flat(),
        y: net.links.map(link => [net.nodes[link.source].y, net.nodes[link.target].y]).flat(),
        mode: 'lines',
        line: { color: '#4b5563', width: 1 },
        type: 'scatter',
        showlegend: false,
        hoverinfo: 'skip',
      },
      // Nodes
      {
        x: net.nodes.map(node => node.x),
        y: net.nodes.map(node => node.y),
        mode: 'markers',
        marker: {
          size: net.nodes.map(node => Math.max(5, Math.min(20, 5 + node.degree))),
          color: net.nodes.map(node => {
            const colors = ['#3b82f6', '#ec4899', '#10b981'];
            return colors[node.group - 1] || '#6b7280';
          }),
          line: { color: '#ffffff', width: 1 }
        },
        text: net.nodes.map(node => `Node ${node.id}<br>Degree: ${node.degree}`),
        type: 'scatter',
        showlegend: false,
      }
    ];

    const layout: Partial<Plotly.Layout> = {
      ...BASE_LAYOUT,
      title: `Scale-Free Network (n=${nodes})`,
      xaxis: { ...BASE_LAYOUT.xaxis, showticklabels: false, showgrid: false },
      yaxis: { ...BASE_LAYOUT.yaxis, showticklabels: false, showgrid: false, scaleanchor: 'x' },
      width: 600,
      height: 500,
    };

    Plotly.newPlot(container, data, layout, { responsive: true, displayModeBar: false });
  }, [nodes, generateScaleFreeNetwork]);

  return (
    <div className="space-y-4">
      <div className="flex gap-4 flex-wrap">
        <div className="flex-1 min-w-32">
          <label className="text-sm text-gray-300 mb-2 block">
            Number of Nodes: {nodes}
          </label>
          <Slider
            value={[nodes]}
            onValueChange={(val) => setNodes(val[0])}
            min={10}
            max={100}
            step={5}
            className="w-full"
          />
        </div>
        <Button onClick={() => setNetwork(generateScaleFreeNetwork(nodes))} className="mt-6">
          Regenerate
        </Button>
      </div>
      <div ref={containerRef} className="w-full h-96 bg-[#151525] rounded-lg overflow-hidden" />
      <p className="text-sm text-gray-400">
        Scale-free network generated using Barabási-Albert preferential attachment model. Node size indicates degree.
      </p>
    </div>
  );
}

// ============ SIMULATION REGISTRY ============

export const COMPLEX_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'percolation': PercolationSimulation,
  'ising-model': IsingModel,
  'scale-free-network': ScaleFreeNetwork,
};

export function getComplexSimulation(id: string): React.ComponentType<SimulationProps> | null {
  return COMPLEX_SIMULATIONS[id] || null;
}