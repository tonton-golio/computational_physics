"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { clamp } from '@/lib/math';
import { SimulationPanel, SimulationSettings, SimulationResults, SimulationLabel, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

type Aggregation = 'mean' | 'sum' | 'max';

interface GnnNode {
  id: string;
  x: number;
  y: number;
  embedding: number[]; // 3D embedding for visualization
}

interface GnnEdge {
  source: string;
  target: string;
}

// Preset graphs
const PRESET_GRAPHS: Record<string, { nodes: Omit<GnnNode, 'embedding'>[]; edges: GnnEdge[] }> = {
  'Small graph': {
    nodes: [
      { id: 'A', x: 150, y: 80 },
      { id: 'B', x: 350, y: 60 },
      { id: 'C', x: 480, y: 180 },
      { id: 'D', x: 400, y: 340 },
      { id: 'E', x: 180, y: 300 },
      { id: 'F', x: 50, y: 190 },
    ],
    edges: [
      { source: 'A', target: 'B' },
      { source: 'A', target: 'F' },
      { source: 'B', target: 'C' },
      { source: 'C', target: 'D' },
      { source: 'D', target: 'E' },
      { source: 'E', target: 'F' },
      { source: 'A', target: 'E' },
      { source: 'B', target: 'D' },
    ],
  },
  'Star graph': {
    nodes: [
      { id: 'Center', x: 280, y: 200 },
      { id: 'A', x: 280, y: 50 },
      { id: 'B', x: 430, y: 120 },
      { id: 'C', x: 430, y: 280 },
      { id: 'D', x: 280, y: 350 },
      { id: 'E', x: 130, y: 280 },
      { id: 'F', x: 130, y: 120 },
    ],
    edges: [
      { source: 'Center', target: 'A' },
      { source: 'Center', target: 'B' },
      { source: 'Center', target: 'C' },
      { source: 'Center', target: 'D' },
      { source: 'Center', target: 'E' },
      { source: 'Center', target: 'F' },
    ],
  },
  'Molecular': {
    nodes: [
      { id: 'C1', x: 200, y: 100 },
      { id: 'C2', x: 350, y: 100 },
      { id: 'C3', x: 425, y: 230 },
      { id: 'C4', x: 350, y: 350 },
      { id: 'C5', x: 200, y: 350 },
      { id: 'C6', x: 125, y: 230 },
      { id: 'O', x: 500, y: 100 },
      { id: 'N', x: 50, y: 100 },
    ],
    edges: [
      { source: 'C1', target: 'C2' },
      { source: 'C2', target: 'C3' },
      { source: 'C3', target: 'C4' },
      { source: 'C4', target: 'C5' },
      { source: 'C5', target: 'C6' },
      { source: 'C6', target: 'C1' },
      { source: 'C2', target: 'O' },
      { source: 'C6', target: 'N' },
    ],
  },
};

function initialEmbedding(nodeId: string, idx: number): number[] {
  // Unique initial embedding per node using simple hash
  const h = nodeId.charCodeAt(0) * 31 + idx * 7;
  return [
    Math.sin(h * 0.1) * 0.5 + 0.5,
    Math.cos(h * 0.3) * 0.5 + 0.5,
    Math.sin(h * 0.7 + 1) * 0.5 + 0.5,
  ];
}

function embeddingToColor(emb: number[]): string {
  const r = Math.round(emb[0] * 200 + 55);
  const g = Math.round(emb[1] * 200 + 55);
  const b = Math.round(emb[2] * 200 + 55);
  return `rgb(${r}, ${g}, ${b})`;
}

function embeddingDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

export default function GnnMessagePassingLive({}: SimulationComponentProps): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [preset, setPreset] = useState<string>('Small graph');
  const [layer, setLayer] = useState(0);
  const [aggregation, setAggregation] = useState<Aggregation>('mean');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [nodes, setNodes] = useState<GnnNode[]>([]);
  const [edges, setEdges] = useState<GnnEdge[]>([]);
  const [_embeddingHistory, setEmbeddingHistory] = useState<number[][][]>([]);

  // Initialize graph from preset
  useEffect(() => {
    const p = PRESET_GRAPHS[preset];
    if (!p) return;
    const newNodes = p.nodes.map((n, i) => ({
      ...n,
      embedding: initialEmbedding(n.id, i),
    }));
    setNodes(newNodes);
    setEdges([...p.edges]);
    setLayer(0);
    setSelectedNode(null);
    setEmbeddingHistory([newNodes.map((n) => [...n.embedding])]);
  }, [preset]);

  // Get neighbors of a node
  const getNeighbors = useCallback(
    (nodeId: string): string[] => {
      const neighbors: string[] = [];
      for (const e of edges) {
        if (e.source === nodeId) neighbors.push(e.target);
        if (e.target === nodeId) neighbors.push(e.source);
      }
      return neighbors;
    },
    [edges],
  );

  // Step one message-passing round
  const step = useCallback(() => {
    setNodes((prevNodes) => {
      const nodeMap = new Map(prevNodes.map((n) => [n.id, n]));
      const newNodes = prevNodes.map((node) => {
        const neighborIds = getNeighbors(node.id);
        if (neighborIds.length === 0) return node;

        const neighborEmbs = neighborIds
          .map((id) => nodeMap.get(id)?.embedding)
          .filter((e): e is number[] => e !== undefined);

        let aggregated: number[];
        switch (aggregation) {
          case 'mean': {
            aggregated = node.embedding.map((_, d) => {
              const sum = neighborEmbs.reduce((a, e) => a + e[d], 0);
              return sum / neighborEmbs.length;
            });
            break;
          }
          case 'sum': {
            aggregated = node.embedding.map((_, d) =>
              neighborEmbs.reduce((a, e) => a + e[d], 0),
            );
            break;
          }
          case 'max': {
            aggregated = node.embedding.map((_, d) =>
              Math.max(...neighborEmbs.map((e) => e[d])),
            );
            break;
          }
        }

        // Update: combine self + aggregated neighbors (simple MLP simulation)
        const newEmb = node.embedding.map((v, d) => {
          const combined = 0.5 * v + 0.5 * aggregated[d];
          return clamp(combined, 0, 1); // ReLU-like clamp
        });

        return { ...node, embedding: newEmb };
      });

      setEmbeddingHistory((prev) => [...prev, newNodes.map((n) => [...n.embedding])]);
      return newNodes;
    });
    setLayer((l) => l + 1);
  }, [aggregation, getNeighbors]);

  const reset = useCallback(() => {
    const p = PRESET_GRAPHS[preset];
    if (!p) return;
    const newNodes = p.nodes.map((n, i) => ({
      ...n,
      embedding: initialEmbedding(n.id, i),
    }));
    setNodes(newNodes);
    setLayer(0);
    setSelectedNode(null);
    setEmbeddingHistory([newNodes.map((n) => [...n.embedding])]);
  }, [preset]);

  // Oversmoothing metric: average pairwise distance between embeddings
  const oversmoothingScore = useMemo(() => {
    if (nodes.length < 2) return 0;
    let totalDist = 0;
    let count = 0;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        totalDist += embeddingDistance(nodes[i].embedding, nodes[j].embedding);
        count++;
      }
    }
    const avgDist = totalDist / count;
    // Normalize: initial avg dist is ~0.5, oversmoothed is ~0
    return clamp(Math.round((1 - avgDist / 0.5) * 100), 0, 100);
  }, [nodes]);

  // Draw graph on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Scale positions to canvas
    const scale = Math.min(W, H) / 550;

    // Draw edges
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
    ctx.lineWidth = 2;
    for (const e of edges) {
      const src = nodes.find((n) => n.id === e.source);
      const dst = nodes.find((n) => n.id === e.target);
      if (!src || !dst) continue;
      ctx.beginPath();
      ctx.moveTo(src.x * scale, src.y * scale);
      ctx.lineTo(dst.x * scale, dst.y * scale);
      ctx.stroke();
    }

    // Highlight selected node's edges
    if (selectedNode) {
      const neighbors = getNeighbors(selectedNode);
      const selNode = nodes.find((n) => n.id === selectedNode);
      if (selNode) {
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 3;
        for (const nId of neighbors) {
          const neighbor = nodes.find((n) => n.id === nId);
          if (!neighbor) continue;
          ctx.beginPath();
          ctx.moveTo(selNode.x * scale, selNode.y * scale);
          ctx.lineTo(neighbor.x * scale, neighbor.y * scale);
          ctx.stroke();

          // Arrow showing message direction
          const dx = selNode.x * scale - neighbor.x * scale;
          const dy = selNode.y * scale - neighbor.y * scale;
          const _dist = Math.sqrt(dx * dx + dy * dy);
          const mx = neighbor.x * scale + dx * 0.3;
          const my = neighbor.y * scale + dy * 0.3;
          const angle = Math.atan2(dy, dx);
          ctx.fillStyle = '#fbbf24';
          ctx.beginPath();
          ctx.moveTo(mx, my);
          ctx.lineTo(mx - 10 * Math.cos(angle - 0.4), my - 10 * Math.sin(angle - 0.4));
          ctx.lineTo(mx - 10 * Math.cos(angle + 0.4), my - 10 * Math.sin(angle + 0.4));
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    // Draw nodes
    const nodeRadius = 24;
    for (const node of nodes) {
      const x = node.x * scale;
      const y = node.y * scale;
      const color = embeddingToColor(node.embedding);

      // Selected highlight
      if (node.id === selectedNode) {
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, nodeRadius + 4, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Node circle
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
      ctx.fill();

      // Border
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label
      ctx.fillStyle = 'white';
      ctx.font = 'bold 13px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.id, x, y);
    }
  }, [nodes, edges, selectedNode, getNeighbors]);

  // Handle click to select node
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const scale = Math.min(canvas.width, canvas.height) / 550;

      for (const node of nodes) {
        const dx = node.x * scale - mx;
        const dy = node.y * scale - my;
        if (dx * dx + dy * dy < 28 * 28) {
          setSelectedNode((prev) => (prev === node.id ? null : node.id));
          return;
        }
      }
      setSelectedNode(null);
    },
    [nodes],
  );

  // Selected node info panel
  const selectedInfo = useMemo(() => {
    if (!selectedNode) return null;
    const node = nodes.find((n) => n.id === selectedNode);
    if (!node) return null;
    const neighbors = getNeighbors(selectedNode);
    return { node, neighbors };
  }, [selectedNode, nodes, getNeighbors]);

  return (
    <SimulationPanel title="GNN Message Passing Live">
      <SimulationSettings>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
          <div>
            <SimulationLabel>Preset</SimulationLabel>
            <select
              value={preset}
              onChange={(e) => setPreset(e.target.value)}
              className="mt-1 w-full rounded bg-[var(--surface-2,#27272a)] px-3 py-1.5 text-sm text-[var(--text-strong)]"
            >
              {Object.keys(PRESET_GRAPHS).map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </div>
          <div>
            <SimulationLabel>Aggregation</SimulationLabel>
            <select
              value={aggregation}
              onChange={(e) => setAggregation(e.target.value as Aggregation)}
              className="mt-1 w-full rounded bg-[var(--surface-2,#27272a)] px-3 py-1.5 text-sm text-[var(--text-strong)]"
            >
              <option value="mean">Mean</option>
              <option value="sum">Sum</option>
              <option value="max">Max</option>
            </select>
          </div>
          <div className="flex items-end gap-2">
            <SimulationButton variant="primary" onClick={step} disabled={layer >= 8}>
              Step (Layer {layer + 1})
            </SimulationButton>
            <SimulationButton variant="secondary" onClick={reset}>
              Reset
            </SimulationButton>
          </div>
        </div>
      </SimulationSettings>

      <SimulationMain scaleMode="contain">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="md:col-span-2">
          <canvas
            ref={canvasRef}
            width={550}
            height={420}
            className="w-full cursor-pointer rounded-xl bg-[var(--surface-2,#27272a)]"
            style={{ maxWidth: 550 }}
            onClick={handleCanvasClick}
          />
          <p className="mt-1 text-xs text-[var(--text-muted)]">
            Click any node to inspect its messages
          </p>
        </div>

        <div className="space-y-4">
          {/* Node info panel */}
          {selectedInfo && (
            <div className="rounded bg-[var(--surface-2,#27272a)] p-3 text-sm">
              <p className="font-medium text-[var(--text-strong)]">
                Node {selectedInfo.node.id}
              </p>
              <p className="mt-1 text-[var(--text-muted)]">
                Embedding: [{selectedInfo.node.embedding.map((v) => v.toFixed(3)).join(', ')}]
              </p>
              <p className="mt-1 text-[var(--text-muted)]">
                Neighbors: {selectedInfo.neighbors.join(', ')}
              </p>
              {layer > 0 && (
                <div className="mt-2">
                  <p className="text-xs font-medium text-[var(--text-muted)]">
                    Messages received ({aggregation}):
                  </p>
                  {selectedInfo.neighbors.map((nId) => {
                    const neighbor = nodes.find((n) => n.id === nId);
                    return neighbor ? (
                      <div key={nId} className="mt-1 flex items-center gap-2">
                        <div
                          className="h-3 w-3 rounded-full"
                          style={{ backgroundColor: embeddingToColor(neighbor.embedding) }}
                        />
                        <span className="text-xs text-[var(--text-muted)]">
                          {nId}: [{neighbor.embedding.map((v) => v.toFixed(2)).join(', ')}]
                        </span>
                      </div>
                    ) : null;
                  })}
                </div>
              )}
            </div>
          )}

          {/* All embeddings table */}
          <div className="rounded bg-[var(--surface-2,#27272a)] p-3 text-sm">
            <p className="mb-2 font-medium text-[var(--text-strong)]">Current Embeddings</p>
            {nodes.map((n) => (
              <div key={n.id} className="flex items-center gap-2 py-0.5">
                <div
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: embeddingToColor(n.embedding) }}
                />
                <span className="w-12 font-mono text-xs text-[var(--text-muted)]">{n.id}</span>
                <span className="font-mono text-xs text-[var(--text-muted)]">
                  [{n.embedding.map((v) => v.toFixed(2)).join(', ')}]
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

        {oversmoothingScore > 70 && (
          <div className="mt-3 rounded bg-amber-900/30 p-3 text-sm text-amber-300">
            Oversmoothing detected ({oversmoothingScore}%). After many message-passing
            layers, all node embeddings converge to similar values — the graph loses its
            ability to distinguish between nodes. This is why most GNNs use 2-3 layers.
          </div>
        )}
      </SimulationMain>
      <SimulationResults>
        <div>
          <SimulationLabel>
            Oversmoothing: {oversmoothingScore}%
          </SimulationLabel>
          <div className="mt-1 h-4 w-full overflow-hidden rounded-full bg-[var(--surface-2,#27272a)]">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${oversmoothingScore}%`,
                backgroundColor: oversmoothingScore > 70 ? '#ef4444' : oversmoothingScore > 40 ? '#f59e0b' : '#10b981',
              }}
            />
          </div>
          <span className="text-xs text-[var(--text-muted)]">Layers applied: {layer}/8</span>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
