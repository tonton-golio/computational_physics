"use client";

import { useState, useMemo } from 'react';
import { SimulationPanel, SimulationSettings, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
  type: 'input' | 'operation' | 'output';
}

interface Edge {
  from: string;
  to: string;
  label?: string;
}

const NODES: Node[] = [
  { id: 'x', label: 'x', x: 50, y: 150, type: 'input' },
  { id: 'w1', label: 'W1', x: 50, y: 50, type: 'input' },
  { id: 'mul1', label: 'W1*x', x: 180, y: 100, type: 'operation' },
  { id: 'b1', label: 'b1', x: 180, y: 200, type: 'input' },
  { id: 'add1', label: '+b1', x: 300, y: 130, type: 'operation' },
  { id: 'relu', label: 'ReLU', x: 420, y: 130, type: 'operation' },
  { id: 'w2', label: 'W2', x: 420, y: 40, type: 'input' },
  { id: 'mul2', label: 'W2*h', x: 540, y: 100, type: 'operation' },
  { id: 'b2', label: 'b2', x: 540, y: 200, type: 'input' },
  { id: 'add2', label: '+b2', x: 650, y: 130, type: 'operation' },
  { id: 'loss', label: 'Loss', x: 770, y: 130, type: 'output' },
];

const EDGES: Edge[] = [
  { from: 'x', to: 'mul1', label: 'x' },
  { from: 'w1', to: 'mul1', label: 'W1' },
  { from: 'mul1', to: 'add1' },
  { from: 'b1', to: 'add1', label: 'b1' },
  { from: 'add1', to: 'relu' },
  { from: 'relu', to: 'mul2', label: 'h' },
  { from: 'w2', to: 'mul2', label: 'W2' },
  { from: 'mul2', to: 'add2' },
  { from: 'b2', to: 'add2', label: 'b2' },
  { from: 'add2', to: 'loss', label: 'logits' },
];

export default function BackpropFlow({}: SimulationComponentProps) {
  const [phase, setPhase] = useState<'forward' | 'backward'>('forward');
  const [step, setStep] = useState(0);

  const maxSteps = phase === 'forward' ? EDGES.length : EDGES.length;

  // Determine active edges/nodes based on step
  const { activeEdges, activeNodes } = useMemo(() => {
    const edges: Set<number> = new Set();
    const nodes: Set<string> = new Set();

    if (phase === 'forward') {
      for (let i = 0; i <= Math.min(step, EDGES.length - 1); i++) {
        edges.add(i);
        nodes.add(EDGES[i].from);
        nodes.add(EDGES[i].to);
      }
    } else {
      // Backward: traverse edges in reverse
      for (let i = EDGES.length - 1; i >= Math.max(EDGES.length - 1 - step, 0); i--) {
        edges.add(i);
        nodes.add(EDGES[i].from);
        nodes.add(EDGES[i].to);
      }
    }

    return { activeEdges: edges, activeNodes: nodes };
  }, [phase, step]);

  const nodeMap = Object.fromEntries(NODES.map(n => [n.id, n]));

  const nodeColor = (node: Node, isActive: boolean) => {
    if (!isActive) return '#333';
    if (node.type === 'input') return '#3b82f6';
    if (node.type === 'output') return '#ef4444';
    return '#8b5cf6';
  };

  const edgeColor = (idx: number) => {
    if (!activeEdges.has(idx)) return '#333';
    return phase === 'forward' ? '#3b82f6' : '#ef4444';
  };

  return (
    <SimulationPanel title="Backpropagation Computational Graph">
      <SimulationSettings>
        <SimulationButton
          variant={phase === 'forward' ? 'primary' : 'secondary'}
          onClick={() => { setPhase('forward'); setStep(0); }}
        >
          Forward Pass
        </SimulationButton>
        <SimulationButton
          variant={phase === 'backward' ? 'danger' : 'secondary'}
          onClick={() => { setPhase('backward'); setStep(0); }}
        >
          Backward Pass
        </SimulationButton>
        <SimulationButton
          onClick={() => setStep(s => Math.max(0, s - 1))}
          disabled={step === 0}
        >
          Prev
        </SimulationButton>
        <SimulationButton
          variant="primary"
          onClick={() => setStep(s => Math.min(maxSteps - 1, s + 1))}
          disabled={step >= maxSteps - 1}
        >
          Next
        </SimulationButton>
        <span className="text-sm text-[var(--text-muted)] flex items-center">
          Step {step + 1}/{maxSteps}
        </span>
      </SimulationSettings>

      <SimulationMain scaleMode="contain">
      <svg viewBox="0 0 830 260" className="w-full h-auto" style={{ maxHeight: 300 }}>
        <defs>
          <marker id="arrow-fwd" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#3b82f6" />
          </marker>
          <marker id="arrow-bwd" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#ef4444" />
          </marker>
          <marker id="arrow-dim" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#333" />
          </marker>
        </defs>

        {/* Edges */}
        {EDGES.map((edge, i) => {
          const from = nodeMap[edge.from];
          const to = nodeMap[edge.to];
          const active = activeEdges.has(i);
          const color = edgeColor(i);
          const isBackward = phase === 'backward' && active;
          const markerEnd = !active ? 'url(#arrow-dim)' : isBackward ? 'url(#arrow-bwd)' : 'url(#arrow-fwd)';

          const x1 = isBackward ? to.x : from.x;
          const y1 = isBackward ? to.y : from.y;
          const x2 = isBackward ? from.x : to.x;
          const y2 = isBackward ? from.y : to.y;

          return (
            <g key={i}>
              <line
                x1={x1 + 30}
                y1={y1}
                x2={x2 - 30}
                y2={y2}
                stroke={color}
                strokeWidth={active ? 2.5 : 1}
                markerEnd={markerEnd}
                opacity={active ? 1 : 0.3}
              />
            </g>
          );
        })}

        {/* Nodes */}
        {NODES.map(node => {
          const isActive = activeNodes.has(node.id);
          const fill = nodeColor(node, isActive);

          return (
            <g key={node.id}>
              <rect
                x={node.x - 30}
                y={node.y - 18}
                width={60}
                height={36}
                rx={8}
                fill={fill}
                opacity={isActive ? 0.9 : 0.2}
                stroke={isActive ? '#fff' : '#555'}
                strokeWidth={isActive ? 1.5 : 0.5}
              />
              <text
                x={node.x}
                y={node.y + 5}
                textAnchor="middle"
                fill={isActive ? '#fff' : '#666'}
                fontSize={12}
                fontWeight={isActive ? 600 : 400}
              >
                {node.label}
              </text>
            </g>
          );
        })}
      </svg>
      </SimulationMain>

      <div className="mt-4 p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        {phase === 'forward' ? (
          <p><strong className="text-blue-300">Forward pass:</strong> Data flows left to right through the computation graph. Each node computes its operation using incoming values, producing activations that feed into the next operation.</p>
        ) : (
          <p><strong className="text-red-300">Backward pass:</strong> Gradients flow right to left using the chain rule. Each node computes the local gradient and multiplies it by the upstream gradient, passing the result to its inputs.</p>
        )}
      </div>
    </SimulationPanel>
  );
}
