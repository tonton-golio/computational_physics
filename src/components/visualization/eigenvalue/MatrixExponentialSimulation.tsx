"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationResults, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { COLORS } from '@/lib/chart-colors';
import type { SimulationComponentProps } from '@/shared/types/simulation';
import { type Matrix2x2, computeUnitEllipse, matrixExp } from './eigen-utils';

export default function MatrixExponentialSimulation({}: SimulationComponentProps) {
  const [matrix, setMatrix] = useState<Matrix2x2>([[1, 0.5], [0, 2]]);

  const { data, layout } = useMemo(() => {
    const expA = matrixExp(matrix);

    // Compute unit circle transformations
    const { circleX, circleY, ellipseX: linearX, ellipseY: linearY } = computeUnitEllipse(matrix);
    const { ellipseX: expX, ellipseY: expY } = computeUnitEllipse(expA);

    const data = [
      // Unit circle
      { x: circleX, y: circleY, type: 'scatter' as const, mode: 'lines' as const, line: { color: '#444', width: 1 }, name: 'Unit circle', showlegend: false },
      // Linear transformation (A)
      { x: linearX, y: linearY, type: 'scatter' as const, mode: 'lines' as const, line: { color: COLORS.primary, width: 2 }, name: 'A \u2022 circle' },
      // Exponential transformation (e^A)
      { x: expX, y: expY, type: 'scatter' as const, mode: 'lines' as const, line: { color: COLORS.secondary, width: 2 }, name: 'e^A \u2022 circle' },
    ];

    const layout = {
      xaxis: { title: { text: 'x' }, range: [-5, 5] },
      yaxis: { title: { text: 'y' }, range: [-5, 5] },
      title: { text: 'Matrix Exponential: Linear vs Exponential Transformation' },
    };

    return { data, layout };
  }, [matrix]);

  return (
    <SimulationPanel title="Matrix Exponential: Linear vs Exponential Transformation" caption="The exponential smooths linear transformations, preserving positivity and orientation.">
      <SimulationSettings>
        <div className="flex gap-4 items-center flex-wrap">
          <span className="text-sm text-[var(--text-soft)]">Matrix A:</span>
          <input
            type="number"
            value={matrix[0][0]}
            onChange={(e) => setMatrix([[parseFloat(e.target.value) || 0, matrix[0][1]], matrix[1]])}
            className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
          />
          <input
            type="number"
            value={matrix[0][1]}
            onChange={(e) => setMatrix([[matrix[0][0], parseFloat(e.target.value) || 0], matrix[1]])}
            className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
          />
          <input
            type="number"
            value={matrix[1][0]}
            onChange={(e) => setMatrix([matrix[0], [parseFloat(e.target.value) || 0, matrix[1][1]]])}
            className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
          />
          <input
            type="number"
            value={matrix[1][1]}
            onChange={(e) => setMatrix([matrix[0], [matrix[1][0], parseFloat(e.target.value) || 0]])}
            className="w-16 px-2 py-1 bg-[var(--surface-1)] rounded text-[var(--text-strong)] text-center"
          />
          <SimulationButton variant="secondary" onClick={() => setMatrix([[1, 0.5], [0, 2]])}>Reset</SimulationButton>
        </div>
      </SimulationSettings>
      <SimulationMain>
        <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 320 }} />
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-soft)] space-y-1">
          <div>e^A ≈ [
            {matrixExp(matrix)[0][0].toFixed(3)}, {matrixExp(matrix)[0][1].toFixed(3)};
            {matrixExp(matrix)[1][0].toFixed(3)}, {matrixExp(matrix)[1][1].toFixed(3)}
          ]</div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
