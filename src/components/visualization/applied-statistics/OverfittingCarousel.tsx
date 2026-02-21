'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

function polyFit(xs: number[], ys: number[], degree: number): number[] {
  const n = xs.length;
  const d = degree + 1;
  // Build Vandermonde matrix
  const X: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < d; j++) row.push(Math.pow(xs[i], j));
    X.push(row);
  }
  // Normal equations: (X^T X) coeffs = X^T y
  const XtX: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const Xty: number[] = new Array(d).fill(0);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      for (let k = 0; k < n; k++) XtX[i][j] += X[k][i] * X[k][j];
    }
    // Ridge: add small regularization to diagonal for numerical stability
    XtX[i][i] += 1e-8;
    for (let k = 0; k < n; k++) Xty[i] += X[k][i] * ys[k];
  }
  // Solve via Gauss elimination
  for (let col = 0; col < d; col++) {
    let maxRow = col;
    for (let row = col + 1; row < d; row++) {
      if (Math.abs(XtX[row][col]) > Math.abs(XtX[maxRow][col])) maxRow = row;
    }
    [XtX[col], XtX[maxRow]] = [XtX[maxRow], XtX[col]];
    [Xty[col], Xty[maxRow]] = [Xty[maxRow], Xty[col]];
    for (let row = col + 1; row < d; row++) {
      const factor = XtX[row][col] / XtX[col][col];
      for (let j = col; j < d; j++) XtX[row][j] -= factor * XtX[col][j];
      Xty[row] -= factor * Xty[col];
    }
  }
  const coeffs = new Array(d).fill(0);
  for (let i = d - 1; i >= 0; i--) {
    coeffs[i] = Xty[i];
    for (let j = i + 1; j < d; j++) coeffs[i] -= XtX[i][j] * coeffs[j];
    coeffs[i] /= XtX[i][i];
  }
  return coeffs;
}

function evalPoly(coeffs: number[], x: number): number {
  let y = 0;
  for (let i = 0; i < coeffs.length; i++) y += coeffs[i] * Math.pow(x, i);
  return y;
}

export default function OverfittingCarousel() {
  const [degree, setDegree] = useState(1);

  const { trainX, trainY, testX, testY, fitX, fitY, trainErr, testErr, degreeErrors } = useMemo(() => {
    // True function: sin(x) + 0.2x
    const trueF = (x: number) => Math.sin(x) + 0.2 * x;
    const nTrain = 15;
    const nTest = 15;

    // Deterministic training data
    const txs: number[] = [];
    const tys: number[] = [];
    for (let i = 0; i < nTrain; i++) {
      const x = i / (nTrain - 1) * 6;
      txs.push(x);
      tys.push(trueF(x) + Math.sin(i * 3.7 + 1.2) * 0.5);
    }

    // Deterministic test data (offset positions)
    const exs: number[] = [];
    const eys: number[] = [];
    for (let i = 0; i < nTest; i++) {
      const x = (i + 0.5) / nTest * 6;
      exs.push(x);
      eys.push(trueF(x) + Math.sin(i * 2.3 + 4.5) * 0.5);
    }

    // Fit polynomial of current degree
    const coeffs = polyFit(txs, tys, degree);
    const fxs: number[] = [];
    const fys: number[] = [];
    for (let i = 0; i <= 100; i++) {
      const x = i / 100 * 6;
      fxs.push(x);
      const y = evalPoly(coeffs, x);
      fys.push(Math.max(-5, Math.min(10, y)));
    }

    const trainMSE = txs.reduce((s, x, i) => s + (tys[i] - evalPoly(coeffs, x)) ** 2, 0) / nTrain;
    const testMSE = exs.reduce((s, x, i) => s + (eys[i] - evalPoly(coeffs, x)) ** 2, 0) / nTest;

    // Compute errors for all degrees
    const errors: { deg: number; train: number; test: number }[] = [];
    for (let d = 1; d <= 15; d++) {
      const c = polyFit(txs, tys, d);
      const tr = txs.reduce((s, x, i) => s + (tys[i] - evalPoly(c, x)) ** 2, 0) / nTrain;
      const te = exs.reduce((s, x, i) => s + (eys[i] - evalPoly(c, x)) ** 2, 0) / nTest;
      errors.push({ deg: d, train: Math.min(tr, 10), test: Math.min(te, 10) });
    }

    return {
      trainX: txs, trainY: tys, testX: exs, testY: eys,
      fitX: fxs, fitY: fys,
      trainErr: trainMSE, testErr: testMSE, degreeErrors: errors,
    };
  }, [degree]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Overfitting: Bias-Variance Tradeoff</h3>
      <div className="grid grid-cols-1 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Polynomial degree: {degree}</label>
          <Slider value={[degree]} onValueChange={([v]) => setDegree(v)} min={1} max={15} step={1} />
        </div>
      </div>
      <div className="mb-3 text-sm text-[var(--text-muted)]">
        Train MSE: {trainErr.toFixed(4)} | Test MSE: {testErr.toFixed(4)}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <CanvasChart
          data={[
            { x: trainX, y: trainY, type: 'scatter', mode: 'markers', marker: { color: '#3b82f6', size: 6 }, name: 'Train' },
            { x: testX, y: testY, type: 'scatter', mode: 'markers', marker: { color: '#f59e0b', size: 6 }, name: 'Test' },
            { x: fitX, y: fitY, type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 2 }, name: `Degree ${degree} fit` },
          ]}
          layout={{
            height: 350,
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'y' }, range: [-3, 8] },
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[
            {
              x: degreeErrors.map((e) => e.deg), y: degreeErrors.map((e) => e.train),
              type: 'scatter', mode: 'lines+markers',
              line: { color: '#3b82f6', width: 2 }, marker: { color: '#3b82f6', size: 5 },
              name: 'Train error',
            },
            {
              x: degreeErrors.map((e) => e.deg), y: degreeErrors.map((e) => e.test),
              type: 'scatter', mode: 'lines+markers',
              line: { color: '#ef4444', width: 2 }, marker: { color: '#ef4444', size: 5 },
              name: 'Test error',
            },
          ]}
          layout={{
            height: 350,
            xaxis: { title: { text: 'Polynomial degree' } },
            yaxis: { title: { text: 'MSE' } },
            shapes: [
              { type: 'line', x0: degree, x1: degree, y0: 0, y1: 10, line: { color: '#94a3b8', width: 1, dash: 'dash' } },
            ],
          }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
