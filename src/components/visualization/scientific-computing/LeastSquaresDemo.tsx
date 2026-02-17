'use client';

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface SimulationProps {
  id?: string;
}

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

export function LeastSquaresDemo({}: SimulationProps) {
  const [nPoints, setNPoints] = useState(40);
  const [noiseLevel, setNoiseLevel] = useState(0.8);
  const [fitDegree, setFitDegree] = useState(1);
  const [seed, setSeed] = useState(42);
  const [showResiduals, setShowResiduals] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animStep, setAnimStep] = useState(0);
  const animRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const generateData = useCallback((s: number) => {
    const rng = seededRandom(s);
    const trueA = 2.5;
    const trueB = 1.0;
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < nPoints; i++) {
      const x = rng() * 4 - 0.5;
      const noise = (rng() - 0.5) * 2 * noiseLevel;
      xs.push(x);
      ys.push(trueA * x + trueB + noise);
    }
    return { xs, ys, trueA, trueB };
  }, [nPoints, noiseLevel]);

  const data = useMemo(() => generateData(seed), [generateData, seed]);

  // Solve normal equations: A^T A c = A^T y
  const fitResult = useMemo(() => {
    const { xs, ys } = data;
    const n = xs.length;
    const deg = fitDegree;

    // Build Vandermonde matrix A (n x (deg+1))
    const A: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j <= deg; j++) {
        row.push(Math.pow(xs[i], j));
      }
      A.push(row);
    }

    // Compute A^T A  ((deg+1) x (deg+1))
    const m = deg + 1;
    const ATA: number[][] = Array.from({ length: m }, () => Array(m).fill(0));
    const ATy: number[] = Array(m).fill(0);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += A[k][i] * A[k][j];
        }
        ATA[i][j] = sum;
      }
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += A[k][i] * ys[k];
      }
      ATy[i] = sum;
    }

    // Solve via Gaussian elimination (small system)
    const augmented = ATA.map((row, i) => [...row, ATy[i]]);
    for (let col = 0; col < m; col++) {
      // Partial pivoting
      let maxRow = col;
      for (let row = col + 1; row < m; row++) {
        if (Math.abs(augmented[row][col]) > Math.abs(augmented[maxRow][col])) {
          maxRow = row;
        }
      }
      [augmented[col], augmented[maxRow]] = [augmented[maxRow], augmented[col]];

      const pivot = augmented[col][col];
      if (Math.abs(pivot) < 1e-12) continue;
      for (let j = col; j <= m; j++) augmented[col][j] /= pivot;
      for (let row = 0; row < m; row++) {
        if (row === col) continue;
        const factor = augmented[row][col];
        for (let j = col; j <= m; j++) {
          augmented[row][j] -= factor * augmented[col][j];
        }
      }
    }

    const coeffs = augmented.map(row => row[m]);

    // Evaluate fit
    const evaluate = (x: number) => {
      let val = 0;
      for (let j = 0; j <= deg; j++) {
        val += coeffs[j] * Math.pow(x, j);
      }
      return val;
    };

    // Compute residuals and R^2
    const yPred = xs.map(evaluate);
    const residuals = ys.map((y, i) => y - yPred[i]);
    const ssRes = residuals.reduce((s, r) => s + r * r, 0);
    const yMean = ys.reduce((s, y) => s + y, 0) / n;
    const ssTot = ys.reduce((s, y) => s + (y - yMean) ** 2, 0);
    const rSquared = 1 - ssRes / ssTot;

    return { coeffs, evaluate, residuals, yPred, rSquared, ssRes };
  }, [data, fitDegree]);

  // Animation: gradient descent approach
  const animResult = useMemo(() => {
    if (!isAnimating && animStep === 0) return null;
    const { xs, ys } = data;
    const n = xs.length;
    let a = 0.5;
    let b = 0.5;
    const lr = 0.02;
    const steps: Array<{ a: number; b: number; loss: number }> = [{ a, b, loss: 0 }];

    for (let step = 0; step < 30; step++) {
      let dLda = 0;
      let dLdb = 0;
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const pred = a * xs[i] + b;
        const err = ys[i] - pred;
        loss += err * err;
        dLda += -2 * err * xs[i];
        dLdb += -2 * err;
      }
      loss /= n;
      dLda /= n;
      dLdb /= n;
      a -= lr * dLda;
      b -= lr * dLdb;
      steps.push({ a, b, loss });
    }
    return steps;
  }, [data, isAnimating, animStep]);

  const handleNewData = useCallback(() => {
    setSeed(prev => prev + 1);
    setAnimStep(0);
    setIsAnimating(false);
  }, []);

  const handleAnimate = useCallback(() => {
    setAnimStep(0);
    setIsAnimating(true);
  }, []);

  useEffect(() => {
    if (!isAnimating) {
      if (animRef.current) clearInterval(animRef.current);
      return;
    }
    animRef.current = setInterval(() => {
      setAnimStep(prev => {
        if (prev >= 29) {
          setIsAnimating(false);
          return 29;
        }
        return prev + 1;
      });
    }, 150);
    return () => {
      if (animRef.current) clearInterval(animRef.current);
    };
  }, [isAnimating]);

  const plotData = useMemo(() => {
    const { xs, ys } = data;
    const traces: Plotly.Data[] = [];

    // Data points
    traces.push({
      x: xs,
      y: ys,
      type: 'scatter' as const,
      mode: 'markers' as const,
      marker: { color: '#3b82f6', size: 7, opacity: 0.8 },
      name: 'Data',
    });

    // Fit curve
    const xFit = Array.from({ length: 200 }, (_, i) => -0.8 + 4.6 * i / 199);
    const yFit = xFit.map(fitResult.evaluate);
    traces.push({
      x: xFit,
      y: yFit,
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#10b981', width: 3 },
      name: `Fit (deg ${fitDegree})`,
    });

    // Animated line
    if (animResult && animStep > 0) {
      const step = animResult[Math.min(animStep, animResult.length - 1)];
      const yAnim = xFit.map(x => step.a * x + step.b);
      traces.push({
        x: xFit,
        y: yAnim,
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#ef4444', width: 2, dash: 'dash' },
        name: `GD step ${animStep}`,
      });
    }

    // Residual lines
    if (showResiduals) {
      for (let i = 0; i < xs.length; i++) {
        traces.push({
          x: [xs[i], xs[i]],
          y: [ys[i], fitResult.yPred[i]],
          type: 'scatter' as const,
          mode: 'lines' as const,
          line: { color: '#f59e0b', width: 1 },
          showlegend: i === 0,
          name: i === 0 ? 'Residuals' : undefined,
        });
      }
    }

    return traces;
  }, [data, fitResult, showResiduals, animResult, animStep, fitDegree]);

  const layout = useMemo(
    () => ({
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(15,15,25,1)',
      font: { color: '#9ca3af', family: 'system-ui' },
      margin: { t: 40, r: 20, b: 50, l: 50 },
      xaxis: {
        title: { text: 'x' },
        gridcolor: '#1e1e2e',
        zerolinecolor: '#2d2d44',
      },
      yaxis: {
        title: { text: 'y' },
        gridcolor: '#1e1e2e',
        zerolinecolor: '#2d2d44',
      },
      title: { text: 'Least Squares Fitting', font: { size: 16 } },
      legend: { x: 0, y: 1, bgcolor: 'rgba(0,0,0,0)' },
      showlegend: true,
    }),
    []
  );

  const coeffString = useMemo(() => {
    const c = fitResult.coeffs;
    if (fitDegree === 1) {
      return `y = ${c[1].toFixed(3)}x + ${c[0].toFixed(3)}`;
    }
    return c
      .map((coeff, i) => {
        if (i === 0) return coeff.toFixed(3);
        if (i === 1) return `${coeff.toFixed(3)}x`;
        return `${coeff.toFixed(3)}x^${i}`;
      })
      .reverse()
      .join(' + ');
  }, [fitResult.coeffs, fitDegree]);

  return (
    <div className="space-y-4">
      <Plot
        data={plotData as Plotly.Data[]}
        layout={layout as Partial<Plotly.Layout>}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: '400px' }}
      />

      <div className="flex flex-wrap gap-4 items-center">
        <label className="text-sm text-gray-400">
          Points:
          <input
            type="range"
            min={10}
            max={100}
            value={nPoints}
            onChange={(e) => setNPoints(parseInt(e.target.value))}
            className="ml-2 w-24 accent-blue-500"
          />
          <span className="ml-1 text-white">{nPoints}</span>
        </label>

        <label className="text-sm text-gray-400">
          Noise:
          <input
            type="range"
            min={0}
            max={3}
            step={0.1}
            value={noiseLevel}
            onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
            className="ml-2 w-24 accent-blue-500"
          />
          <span className="ml-1 text-white">{noiseLevel.toFixed(1)}</span>
        </label>

        <label className="text-sm text-gray-400">
          Degree:
          <select
            value={fitDegree}
            onChange={(e) => setFitDegree(parseInt(e.target.value))}
            className="ml-2 bg-[#151525] text-white rounded px-2 py-1"
          >
            <option value={1}>1 (Linear)</option>
            <option value={2}>2 (Quadratic)</option>
            <option value={3}>3 (Cubic)</option>
            <option value={4}>4 (Quartic)</option>
          </select>
        </label>
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleNewData}
          className="px-4 py-2 bg-blue-600 rounded text-sm hover:bg-blue-700 text-white"
        >
          New Data
        </button>
        <button
          onClick={handleAnimate}
          className="px-4 py-2 bg-green-600 rounded text-sm hover:bg-green-700 text-white"
        >
          Animate GD
        </button>
        <button
          onClick={() => setShowResiduals(!showResiduals)}
          className={`px-4 py-2 rounded text-sm text-white ${showResiduals ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-gray-600 hover:bg-gray-700'}`}
        >
          {showResiduals ? 'Hide' : 'Show'} Residuals
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
        <div className="bg-[#151525] rounded-lg p-3">
          <div className="text-gray-400">Fit equation</div>
          <div className="text-white font-mono text-xs mt-1">{coeffString}</div>
        </div>
        <div className="bg-[#151525] rounded-lg p-3">
          <div className="text-gray-400">R-squared</div>
          <div className="text-green-400 font-mono mt-1">{fitResult.rSquared.toFixed(6)}</div>
        </div>
        <div className="bg-[#151525] rounded-lg p-3">
          <div className="text-gray-400">Sum of Squared Residuals</div>
          <div className="text-yellow-400 font-mono mt-1">{fitResult.ssRes.toFixed(4)}</div>
        </div>
      </div>

      <p className="text-xs text-gray-500">
        The fit is computed via the normal equations: solve A^T A c = A^T y where A is the Vandermonde matrix.
        Press &quot;Animate GD&quot; to watch gradient descent converge to the linear fit from a random initial guess.
      </p>
    </div>
  );
}
