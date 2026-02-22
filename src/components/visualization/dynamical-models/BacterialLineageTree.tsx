"use client";

import { useState, useMemo } from 'react';
import { mulberry32 } from '@/lib/math';
import { Slider } from '@/components/ui/slider';
import { SimulationMain } from '@/components/ui/simulation-main';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * BacterialLineageTree: Visualizes a bacterial population dividing over
 * generations with stochastic partitioning of molecules at division.
 *
 * Each cell starts with some number of molecules. At division, each molecule
 * goes to one daughter with probability 0.5 (binomial partitioning).
 * The tree shows how molecule counts vary across a growing population.
 */

interface CellNode {
  x: number;
  y: number;
  molecules: number;
  children: [CellNode, CellNode] | null;
}

function buildTree(initialMolecules: number, generations: number, seed: number): CellNode {
  const rand = mulberry32(seed);

  function seededBinomialSplit(n: number): [number, number] {
    let left = 0;
    for (let i = 0; i < n; i++) {
      if (rand() < 0.5) left++;
    }
    return [left, n - left];
  }

  function buildNode(molecules: number, gen: number, xCenter: number, yLevel: number, xSpan: number): CellNode {
    if (gen >= generations) {
      return { x: xCenter, y: yLevel, molecules, children: null };
    }

    const [leftMol, rightMol] = seededBinomialSplit(molecules);
    const childSpan = xSpan / 2;

    const leftChild = buildNode(leftMol, gen + 1, xCenter - childSpan / 2, yLevel + 1, childSpan);
    const rightChild = buildNode(rightMol, gen + 1, xCenter + childSpan / 2, yLevel + 1, childSpan);

    return { x: xCenter, y: yLevel, molecules, children: [leftChild, rightChild] };
  }

  return buildNode(initialMolecules, 0, 0.5, 0, 1);
}

function collectNodes(node: CellNode): CellNode[] {
  const nodes: CellNode[] = [node];
  if (node.children) {
    nodes.push(...collectNodes(node.children[0]));
    nodes.push(...collectNodes(node.children[1]));
  }
  return nodes;
}

function collectEdges(node: CellNode): Array<{ parent: CellNode; child: CellNode }> {
  const edges: Array<{ parent: CellNode; child: CellNode }> = [];
  if (node.children) {
    edges.push({ parent: node, child: node.children[0] });
    edges.push({ parent: node, child: node.children[1] });
    edges.push(...collectEdges(node.children[0]));
    edges.push(...collectEdges(node.children[1]));
  }
  return edges;
}

function moleculeColor(molecules: number, maxMol: number): string {
  const t = Math.min(molecules / Math.max(maxMol, 1), 1);
  // Interpolate from red (low) to blue (high)
  const r = Math.round(239 * (1 - t) + 59 * t);
  const g = Math.round(68 * (1 - t) + 130 * t);
  const b = Math.round(68 * (1 - t) + 246 * t);
  return `rgb(${r},${g},${b})`;
}

export default function BacterialLineageTree({}: SimulationComponentProps) {
  const [initialMolecules, setInitialMolecules] = useState(40);
  const [generations, setGenerations] = useState(5);
  const [seed, setSeed] = useState(42);

  const tree = useMemo(
    () => buildTree(initialMolecules, generations, seed),
    [initialMolecules, generations, seed],
  );

  const allNodes = useMemo(() => collectNodes(tree), [tree]);
  const allEdges = useMemo(() => collectEdges(tree), [tree]);
  const maxMol = useMemo(() => Math.max(...allNodes.map(n => n.molecules), 1), [allNodes]);

  const leafNodes = useMemo(
    () => allNodes.filter(n => n.children === null),
    [allNodes],
  );

  const stats = useMemo(() => {
    const mols = leafNodes.map(n => n.molecules);
    const mean = mols.reduce((a, b) => a + b, 0) / mols.length;
    const variance = mols.reduce((a, b) => a + (b - mean) ** 2, 0) / mols.length;
    const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
    return { mean, variance, cv, min: Math.min(...mols), max: Math.max(...mols) };
  }, [leafNodes]);

  // SVG dimensions
  const svgW = 800;
  const svgH = 80 + generations * 80;
  const padX = 40;
  const padY = 40;

  const scaleX = (x: number) => padX + x * (svgW - 2 * padX);
  const scaleY = (y: number) => padY + y * (svgH - 2 * padY) / generations;

  const nodeRadius = Math.max(6, 18 - generations * 2);

  return (
    <SimulationPanel title="Bacterial Lineage Tree: Stochastic Partitioning">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Initial molecules: {initialMolecules}
          </SimulationLabel>
          <Slider value={[initialMolecules]} onValueChange={([v]) => setInitialMolecules(v)} min={4} max={100} step={2} />
        </div>
        <div>
          <SimulationLabel>
            Generations: {generations}
          </SimulationLabel>
          <Slider value={[generations]} onValueChange={([v]) => setGenerations(v)} min={2} max={7} step={1} />
        </div>
        <div>
          <SimulationLabel>
            Random seed: {seed}
          </SimulationLabel>
          <Slider value={[seed]} onValueChange={([v]) => setSeed(v)} min={1} max={100} step={1} />
        </div>
      </SimulationConfig>

      <SimulationMain scaleMode="contain" className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${svgW} ${svgH}`}
          className="w-full"
          style={{ maxHeight: 500, minWidth: 400 }}
        >
          {/* Edges */}
          {allEdges.map((edge, i) => (
            <line
              key={`e-${i}`}
              x1={scaleX(edge.parent.x)}
              y1={scaleY(edge.parent.y)}
              x2={scaleX(edge.child.x)}
              y2={scaleY(edge.child.y)}
              stroke="var(--border-strong)"
              strokeWidth={1.5}
              opacity={0.5}
            />
          ))}

          {/* Nodes */}
          {allNodes.map((node, i) => (
            <g key={`n-${i}`}>
              <circle
                cx={scaleX(node.x)}
                cy={scaleY(node.y)}
                r={nodeRadius}
                fill={moleculeColor(node.molecules, maxMol)}
                stroke="var(--border-strong)"
                strokeWidth={1}
              />
              {nodeRadius >= 10 && (
                <text
                  x={scaleX(node.x)}
                  y={scaleY(node.y)}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={nodeRadius > 12 ? 10 : 8}
                  fill="white"
                  fontWeight="bold"
                >
                  {node.molecules}
                </text>
              )}
            </g>
          ))}

          {/* Generation labels */}
          {Array.from({ length: generations + 1 }, (_, g) => (
            <text
              key={`gen-${g}`}
              x={12}
              y={scaleY(g)}
              textAnchor="start"
              dominantBaseline="central"
              fontSize={11}
              fill="var(--text-muted)"
            >
              G{g}
            </text>
          ))}
        </svg>
      </SimulationMain>

      <SimulationResults>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Leaf mean</div>
          <div className="text-lg font-mono">{stats.mean.toFixed(1)}</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Leaf CV</div>
          <div className="text-lg font-mono">{stats.cv.toFixed(3)}</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Min / Max</div>
          <div className="text-lg font-mono">{stats.min} / {stats.max}</div>
        </div>
        <div className="bg-[var(--surface-2)] rounded-md p-3">
          <div className="font-medium text-[var(--text-strong)]">Expected CV</div>
          <div className="text-lg font-mono">{(1 / Math.sqrt(initialMolecules)).toFixed(3)}</div>
          <div className="text-xs text-[var(--text-muted)]">1/&radic;N theory</div>
        </div>
      </div>
      </SimulationResults>

      <div className="mt-4 border-l-4 border-blue-500 pl-4 text-sm text-[var(--text-muted)]">
        <p className="font-medium text-[var(--text-strong)] mb-1">What to notice</p>
        <p>
          Each division partitions molecules binomially between the two daughters.
          With few molecules, the noise is dramatic &mdash; some lineages end up nearly
          empty while others are rich. The coefficient of variation (CV) at the leaves
          scales as 1/&radic;N for a single division, showing why low-copy-number
          molecules (like mRNA) create large cell-to-cell variability.
          Try reducing initial molecules to 4&ndash;10 to see the effect.
        </p>
      </div>
    </SimulationPanel>
  );
}
