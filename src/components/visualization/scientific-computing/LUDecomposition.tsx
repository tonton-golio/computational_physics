'use client';

import { useState, useMemo, useCallback } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id?: string;
}

type Matrix = number[][];

function formatNum(n: number): string {
  if (Number.isInteger(n)) return n.toString();
  return n.toFixed(3);
}

export function LUDecomposition({}: SimulationProps) {
  const [matrixInput, setMatrixInput] = useState<Matrix>([
    [1, 4, 12],
    [5, 4, 2],
    [9, 5, 67],
  ]);
  const [rhsInput, setRhsInput] = useState<number[]>([1, 2, 3]);
  const [currentStep, setCurrentStep] = useState(0);
  const [showSolve, setShowSolve] = useState(false);

  // LU factorization with step tracking
  const luSteps = useMemo(() => {
    const n = matrixInput.length;
    const U: Matrix = matrixInput.map(row => [...row]);
    const L: Matrix = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
    );

    const steps: Array<{
      description: string;
      L: Matrix;
      U: Matrix;
      highlightRow?: number;
      highlightCol?: number;
      scalar?: number;
    }> = [];

    // Initial state
    steps.push({
      description: 'Start: U = A, L = I',
      L: L.map(r => [...r]),
      U: U.map(r => [...r]),
    });

    for (let j = 0; j < n - 1; j++) {
      for (let i = j + 1; i < n; i++) {
        if (Math.abs(U[j][j]) < 1e-12) {
          steps.push({
            description: `Pivot U[${j}][${j}] is zero -- cannot proceed without pivoting.`,
            L: L.map(r => [...r]),
            U: U.map(r => [...r]),
            highlightRow: i,
            highlightCol: j,
          });
          continue;
        }
        const scalar = U[i][j] / U[j][j];
        L[i][j] = scalar;

        steps.push({
          description: `Compute scalar: L[${i}][${j}] = U[${i}][${j}] / U[${j}][${j}] = ${formatNum(U[i][j])} / ${formatNum(U[j][j])} = ${formatNum(scalar)}`,
          L: L.map(r => [...r]),
          U: U.map(r => [...r]),
          highlightRow: i,
          highlightCol: j,
          scalar,
        });

        for (let k = 0; k < n; k++) {
          U[i][k] -= scalar * U[j][k];
        }

        steps.push({
          description: `Eliminate: Row ${i} = Row ${i} - ${formatNum(scalar)} * Row ${j}`,
          L: L.map(r => [...r]),
          U: U.map(r => [...r]),
          highlightRow: i,
          highlightCol: j,
        });
      }
    }

    steps.push({
      description: 'LU Factorization complete!',
      L: L.map(r => [...r]),
      U: U.map(r => [...r]),
    });

    return steps;
  }, [matrixInput]);

  // Forward / backward substitution
  const solveResult = useMemo(() => {
    if (!showSolve) return null;
    const finalStep = luSteps[luSteps.length - 1];
    const L = finalStep.L;
    const U = finalStep.U;
    const b = [...rhsInput];
    const n = L.length;

    // Forward substitution: Ly = b
    const y = new Array(n).fill(0);
    const forwardSteps: string[] = [];
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < i; j++) {
        sum += L[i][j] * y[j];
      }
      y[i] = (b[i] - sum) / L[i][i];
      forwardSteps.push(`y[${i}] = (b[${i}] - sum) / L[${i}][${i}] = (${formatNum(b[i])} - ${formatNum(sum)}) / ${formatNum(L[i][i])} = ${formatNum(y[i])}`);
    }

    // Backward substitution: Ux = y
    const x = new Array(n).fill(0);
    const backwardSteps: string[] = [];
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) {
        sum += U[i][j] * x[j];
      }
      x[i] = (y[i] - sum) / U[i][i];
      backwardSteps.push(`x[${i}] = (y[${i}] - sum) / U[${i}][${i}] = (${formatNum(y[i])} - ${formatNum(sum)}) / ${formatNum(U[i][i])} = ${formatNum(x[i])}`);
    }

    // Verification: A * x
    const Ax = matrixInput.map(row => row.reduce((s, v, j) => s + v * x[j], 0));

    return { y, x, forwardSteps, backwardSteps, Ax };
  }, [showSolve, luSteps, rhsInput, matrixInput]);

  const step = luSteps[Math.min(currentStep, luSteps.length - 1)];

  const handleNextStep = useCallback(() => {
    setCurrentStep(prev => Math.min(prev + 1, luSteps.length - 1));
  }, [luSteps.length]);

  const handlePrevStep = useCallback(() => {
    setCurrentStep(prev => Math.max(prev - 1, 0));
  }, []);

  const handleResetMatrix = useCallback(() => {
    setMatrixInput([
      [1, 4, 12],
      [5, 4, 2],
      [9, 5, 67],
    ]);
    setRhsInput([1, 2, 3]);
    setCurrentStep(0);
    setShowSolve(false);
  }, []);

  // Heatmap data for L and U
  const makeHeatmap = useCallback((mat: Matrix, title: string, colorscale: string) => {
    const n = mat.length;
    const annotations = mat.flatMap((row, i) =>
      row.map((val, j) => ({
        x: j,
        y: i,
        text: formatNum(val),
        showarrow: false,
        font: { color: Math.abs(val) > 3 ? '#fff' : '#e5e7eb', size: 13 },
      }))
    );

    return {
      data: [
        {
          z: mat,
          type: 'heatmap' as const,
          colorscale,
          showscale: false,
          x: Array.from({ length: n }, (_, i) => i),
          y: Array.from({ length: n }, (_, i) => i),
        },
      ],
      layout: {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15,15,25,1)',
        font: { color: '#9ca3af', family: 'system-ui' },
        margin: { t: 35, r: 10, b: 10, l: 10 },
        title: { text: title, font: { size: 14 } },
        xaxis: { showticklabels: false, ticks: '' as const, showgrid: false },
        yaxis: { showticklabels: false, ticks: '' as const, showgrid: false, autorange: 'reversed' as const },
        annotations,
        width: 220,
        height: 220,
      },
    };
  }, []);

  const lHeatmap = useMemo(() => makeHeatmap(step.L, 'L (Lower)', 'Blues'), [step.L, makeHeatmap]);
  const uHeatmap = useMemo(() => makeHeatmap(step.U, 'U (Upper)', 'Reds'), [step.U, makeHeatmap]);

  return (
    <div className="space-y-4">
      {/* Matrix input */}
      <div className="flex flex-wrap gap-6 items-start">
        <div>
          <div className="text-sm text-gray-400 mb-2">Matrix A:</div>
          <div className="grid grid-cols-3 gap-1">
            {matrixInput.map((row, i) =>
              row.map((val, j) => (
                <input
                  key={`${i}-${j}`}
                  type="number"
                  step="1"
                  value={val}
                  onChange={(e) => {
                    const newM = matrixInput.map(r => [...r]);
                    newM[i][j] = parseFloat(e.target.value) || 0;
                    setMatrixInput(newM);
                    setCurrentStep(0);
                    setShowSolve(false);
                  }}
                  className="w-16 px-2 py-1 bg-[#151525] rounded text-white text-center text-sm"
                />
              ))
            )}
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-400 mb-2">RHS b:</div>
          <div className="grid grid-cols-1 gap-1">
            {rhsInput.map((val, i) => (
              <input
                key={i}
                type="number"
                step="1"
                value={val}
                onChange={(e) => {
                  const newB = [...rhsInput];
                  newB[i] = parseFloat(e.target.value) || 0;
                  setRhsInput(newB);
                  setShowSolve(false);
                }}
                className="w-16 px-2 py-1 bg-[#151525] rounded text-white text-center text-sm"
              />
            ))}
          </div>
        </div>
      </div>

      {/* LU Heatmaps */}
      <div className="flex flex-wrap gap-4 justify-center">
        <Plot
          data={lHeatmap.data as Plotly.Data[]}
          layout={lHeatmap.layout as Partial<Plotly.Layout>}
          config={{ responsive: false, displayModeBar: false }}
        />
        <Plot
          data={uHeatmap.data as Plotly.Data[]}
          layout={uHeatmap.layout as Partial<Plotly.Layout>}
          config={{ responsive: false, displayModeBar: false }}
        />
      </div>

      {/* Step description */}
      <div className="bg-[#151525] rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-1">
          Step {currentStep + 1} / {luSteps.length}
        </div>
        <div className="text-white font-mono text-sm">{step.description}</div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={handlePrevStep}
          disabled={currentStep === 0}
          className="px-4 py-2 bg-gray-600 rounded text-sm hover:bg-gray-700 text-white disabled:opacity-40"
        >
          Previous
        </button>
        <button
          onClick={handleNextStep}
          disabled={currentStep >= luSteps.length - 1}
          className="px-4 py-2 bg-blue-600 rounded text-sm hover:bg-blue-700 text-white disabled:opacity-40"
        >
          Next Step
        </button>
        <button
          onClick={() => setCurrentStep(luSteps.length - 1)}
          className="px-4 py-2 bg-purple-600 rounded text-sm hover:bg-purple-700 text-white"
        >
          Skip to End
        </button>
        <button
          onClick={() => { setShowSolve(true); setCurrentStep(luSteps.length - 1); }}
          className="px-4 py-2 bg-green-600 rounded text-sm hover:bg-green-700 text-white"
        >
          Solve Ax = b
        </button>
        <button
          onClick={handleResetMatrix}
          className="px-4 py-2 bg-gray-600 rounded text-sm hover:bg-gray-700 text-white"
        >
          Reset
        </button>
      </div>

      {/* Solve result */}
      {solveResult && (
        <div className="space-y-3">
          <div className="bg-[#0f0f1f] rounded-lg p-4 space-y-2">
            <div className="text-sm text-gray-400 font-semibold">Forward Substitution (Ly = b):</div>
            {solveResult.forwardSteps.map((s, i) => (
              <div key={i} className="text-xs font-mono text-blue-300">{s}</div>
            ))}
            <div className="text-sm text-white mt-2">
              y = [{solveResult.y.map(v => formatNum(v)).join(', ')}]
            </div>
          </div>

          <div className="bg-[#0f0f1f] rounded-lg p-4 space-y-2">
            <div className="text-sm text-gray-400 font-semibold">Backward Substitution (Ux = y):</div>
            {solveResult.backwardSteps.map((s, i) => (
              <div key={i} className="text-xs font-mono text-green-300">{s}</div>
            ))}
            <div className="text-sm text-white mt-2">
              x = [{solveResult.x.map(v => formatNum(v)).join(', ')}]
            </div>
          </div>

          <div className="bg-[#0f0f1f] rounded-lg p-4">
            <div className="text-sm text-gray-400 font-semibold">Verification: Ax = </div>
            <div className="text-sm text-yellow-300 font-mono">
              [{solveResult.Ax.map(v => formatNum(v)).join(', ')}]
            </div>
            <div className="text-sm text-gray-400">
              b = [{rhsInput.map(v => formatNum(v)).join(', ')}]
            </div>
          </div>
        </div>
      )}

      <p className="text-xs text-gray-500">
        Step through the LU factorization to see how each elimination creates an entry in L while zeroing
        the corresponding entry in U. Then solve Ax = b via forward substitution (Ly = b) followed
        by backward substitution (Ux = y).
      </p>
    </div>
  );
}
