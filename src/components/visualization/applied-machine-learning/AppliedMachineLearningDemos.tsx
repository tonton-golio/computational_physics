'use client';

import React, { useMemo, useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { linspace, pseudoRandom, gaussianNoise } from './ml-utils';

function card(title: string, children: React.ReactNode): React.ReactElement {
  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6">
      <h3 className="mb-4 text-xl font-semibold text-[var(--text-strong)]">{title}</h3>
      {children}
    </div>
  );
}

export function LossFunctionsDemo(): React.ReactElement {
  const [margin, setMargin] = useState(1);
  const x = useMemo(() => linspace(-3, 3, 200), []);
  const { zeroOne, hinge, bce, mse } = useMemo(() => {
    const z = x.map((v) => (v < 0 ? 1 : 0));
    const h = x.map((v) => Math.max(0, margin - v));
    const p = x.map((v) => 1 / (1 + Math.exp(-v)));
    const b = p.map((pv) => -Math.log(Math.max(pv, 1e-6)));
    const m = x.map((v) => v * v);
    return { zeroOne: z, hinge: h, bce: b, mse: m };
  }, [x, margin]);

  return card(
    'Loss Functions',
    <>
      <div className="mb-4">
        <label className="text-sm text-[var(--text-muted)]">Hinge margin: {margin.toFixed(2)}</label>
        <Slider
          min={0.2}
          max={2}
          step={0.05}
          value={[margin]}
          onValueChange={([v]) => setMargin(v)}
        />
      </div>
      <CanvasChart
        data={[
          { x, y: zeroOne, type: 'scatter', mode: 'lines', name: 'Zero-One' },
          { x, y: hinge, type: 'scatter', mode: 'lines', name: 'Hinge' },
          { x, y: bce, type: 'scatter', mode: 'lines', name: 'BCE (y=1)' },
          { x, y: mse, type: 'scatter', mode: 'lines', name: 'MSE' },
        ]}
        layout={{
          xaxis: { title: { text: 'Score/Residual' } },
          yaxis: { title: { text: 'Loss' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function LossLandscapeDemo(): React.ReactElement {
  const [noise, setNoise] = useState(0.2);
  const [ripple, setRipple] = useState(2.5);
  const grid = useMemo(() => {
    const x = linspace(-3, 3, 45);
    const y = linspace(-3, 3, 45);
    const z = y.map((yy) =>
      x.map((xx, j) => {
        const seed = (yy + 7.1) * 1000 + j * 17.3 + ripple * 31;
        return xx * xx + 0.8 * yy * yy + 0.5 * Math.sin(ripple * xx) + gaussianNoise(seed, noise);
      })
    );
    return { x, y, z };
  }, [noise, ripple]);

  return card(
    'Loss Landscape',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Noise: {noise.toFixed(2)}</label>
          <Slider
            min={0}
            max={0.8}
            step={0.05}
            value={[noise]}
            onValueChange={([v]) => setNoise(v)}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Ripple frequency: {ripple.toFixed(1)}</label>
          <Slider
            min={0.5}
            max={5}
            step={0.1}
            value={[ripple]}
            onValueChange={([v]) => setRipple(v)}
          />
        </div>
      </div>
      <CanvasHeatmap
        data={[
          {
            z: grid.z,
            colorscale: 'Viridis',
            showscale: false,
          },
        ]}
        layout={{
          xaxis: { title: { text: 'w1' } },
          yaxis: { title: { text: 'w2' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 420 }}
      />
    </>
  );
}

export function ValidationSplitDemo(): React.ReactElement {
  const [epochs, setEpochs] = useState(50);
  const [trainPct, setTrainPct] = useState(60);
  const [valPct, setValPct] = useState(20);
  const testPct = 100 - trainPct - valPct;
  const overfitStart = useMemo(() => Math.max(8, Math.round(35 * (trainPct / 80))), [trainPct]);

  // Less training data = noisier curves and earlier overfitting
  const noiseFactor = useMemo(() => Math.max(0.5, (80 - trainPct) / 40), [trainPct]);

  const curves = useMemo(() => {
    const x = Array.from({ length: epochs }, (_, i) => i + 1);
    const train = x.map((e) => {
      const base = Math.exp(-e / 11);
      const noise = 0.03 * noiseFactor * Math.sin(e / 2 + trainPct * 0.1);
      return base + noise;
    });
    const val = x.map((e) => {
      const base = Math.exp(-e / 10);
      const penalty = e > overfitStart ? 0.003 * noiseFactor * (e - overfitStart) ** 2 : 0;
      const noise = 0.04 * noiseFactor * Math.cos(e / 3 + valPct * 0.1);
      return base + 0.07 * noiseFactor + penalty + noise;
    });
    const test = x.map((e) => {
      const base = Math.exp(-e / 10);
      const penalty = e > overfitStart ? 0.004 * noiseFactor * (e - overfitStart) ** 2 : 0;
      return base + 0.12 * noiseFactor + penalty;
    });
    return { x, train, val, test };
  }, [epochs, trainPct, valPct, overfitStart, noiseFactor]);

  return card(
    'Train / Validation / Test Split',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-4">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Train: {trainPct}%</label>
          <Slider
            min={30}
            max={85}
            step={5}
            value={[trainPct]}
            onValueChange={([v]) => {
              const remaining = 100 - v;
              setTrainPct(v);
              if (valPct > remaining - 5) setValPct(Math.max(5, remaining - 5));
            }}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Val: {valPct}%</label>
          <Slider
            min={5}
            max={Math.max(5, 100 - trainPct - 5)}
            step={5}
            value={[valPct]}
            onValueChange={([v]) => setValPct(v)}
          />
        </div>
        <div className="flex items-end">
          <span className="rounded bg-[var(--surface-2,#27272a)] px-3 py-1.5 text-sm text-[var(--text-strong)]">
            Test: {testPct}%
          </span>
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Epochs: {epochs}</label>
          <Slider min={15} max={80} step={1} value={[epochs]} onValueChange={([v]) => setEpochs(v)} />
        </div>
      </div>
      {/* Visual split bar */}
      <div className="mb-4 flex h-6 w-full overflow-hidden rounded-full text-xs font-medium">
        <div className="flex items-center justify-center bg-blue-600 text-white" style={{ width: `${trainPct}%` }}>
          Train
        </div>
        <div className="flex items-center justify-center bg-amber-500 text-white" style={{ width: `${valPct}%` }}>
          Val
        </div>
        <div className="flex items-center justify-center bg-emerald-600 text-white" style={{ width: `${testPct}%` }}>
          Test
        </div>
      </div>
      <CanvasChart
        data={[
          { x: curves.x, y: curves.train, type: 'scatter', mode: 'lines', name: 'Training loss', line: { color: '#3b82f6', width: 2 } },
          { x: curves.x, y: curves.val, type: 'scatter', mode: 'lines', name: 'Validation loss', line: { color: '#f59e0b', width: 2 } },
          { x: curves.x, y: curves.test, type: 'scatter', mode: 'lines', name: 'Test loss', line: { color: '#10b981', width: 2 } },
          // Early stopping line
          { x: [overfitStart, overfitStart], y: [0, 1.2], type: 'scatter', mode: 'lines', name: 'Early stop', line: { color: '#ef4444', width: 1 } },
        ]}
        layout={{
          xaxis: { title: { text: 'Epoch' } },
          yaxis: { title: { text: 'Loss' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 360 }}
      />
      {trainPct < 45 && (
        <div className="mt-2 text-sm text-amber-400">
          With only {trainPct}% training data, curves are noisy and overfitting starts earlier.
        </div>
      )}
    </>
  );
}

export function TreeSplitImpurityDemo(): React.ReactElement {
  const [threshold, setThreshold] = useState(0.5);
  const [splitFeature, setSplitFeature] = useState<0 | 1>(0);

  // Generate a 2D dataset with two clusters
  const data = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    const labels: number[] = [];
    for (let i = 0; i < 150; i++) {
      const c = i < 75 ? 0 : 1;
      xs.push((c === 0 ? 0.3 : 0.7) + gaussianNoise(i * 2.1 + 5, 0.15));
      ys.push((c === 0 ? 0.35 : 0.65) + gaussianNoise(i * 3.3 + 11, 0.15));
      labels.push(c);
    }
    return { xs, ys, labels };
  }, []);

  // Compute impurity for the current split
  const { leftGini, rightGini, leftEntropy, rightEntropy, weightedGini, weightedEntropy, leftCount, rightCount } =
    useMemo(() => {
      const featureVals = splitFeature === 0 ? data.xs : data.ys;
      const leftLabels: number[] = [];
      const rightLabels: number[] = [];
      for (let i = 0; i < featureVals.length; i++) {
        if (featureVals[i] <= threshold) leftLabels.push(data.labels[i]);
        else rightLabels.push(data.labels[i]);
      }
      const gini = (ls: number[]) => {
        if (ls.length === 0) return 0;
        const p = ls.reduce((a, b) => a + b, 0) / ls.length;
        return 1 - p * p - (1 - p) * (1 - p);
      };
      const ent = (ls: number[]) => {
        if (ls.length === 0) return 0;
        const p = ls.reduce((a, b) => a + b, 0) / ls.length;
        if (p < 1e-9 || p > 1 - 1e-9) return 0;
        return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
      };
      const n = featureVals.length;
      const lg = gini(leftLabels);
      const rg = gini(rightLabels);
      const le = ent(leftLabels);
      const re = ent(rightLabels);
      return {
        leftGini: lg,
        rightGini: rg,
        leftEntropy: le,
        rightEntropy: re,
        weightedGini: (leftLabels.length / n) * lg + (rightLabels.length / n) * rg,
        weightedEntropy: (leftLabels.length / n) * le + (rightLabels.length / n) * re,
        leftCount: leftLabels.length,
        rightCount: rightLabels.length,
      };
    }, [data, threshold, splitFeature]);

  const colors = data.labels.map((l) => (l === 0 ? '#3b82f6' : '#ef4444'));
  const featureLabel = splitFeature === 0 ? 'x\u2081' : 'x\u2082';

  return card(
    'Tree Split & Impurity',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <label className="text-sm text-[var(--text-muted)]">
            Split threshold ({featureLabel}): {threshold.toFixed(2)}
          </label>
          <Slider min={0} max={1} step={0.01} value={[threshold]} onValueChange={([v]) => setThreshold(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Split feature</label>
          <div className="mt-1 flex gap-2">
            <button
              onClick={() => setSplitFeature(0)}
              className={`rounded px-3 py-1 text-xs ${splitFeature === 0 ? 'bg-[var(--accent,#3b82f6)] text-white' : 'bg-[var(--surface-2,#27272a)] text-[var(--text-strong)]'}`}
            >
              x&#8321;
            </button>
            <button
              onClick={() => setSplitFeature(1)}
              className={`rounded px-3 py-1 text-xs ${splitFeature === 1 ? 'bg-[var(--accent,#3b82f6)] text-white' : 'bg-[var(--surface-2,#27272a)] text-[var(--text-strong)]'}`}
            >
              x&#8322;
            </button>
          </div>
        </div>
        <div className="text-sm text-[var(--text-muted)]">
          <div>Left ({leftCount}): Gini={leftGini.toFixed(3)}, H={leftEntropy.toFixed(3)}</div>
          <div>Right ({rightCount}): Gini={rightGini.toFixed(3)}, H={rightEntropy.toFixed(3)}</div>
          <div className="mt-1 font-medium text-[var(--text-strong)]">
            Weighted Gini: {weightedGini.toFixed(4)} | Entropy: {weightedEntropy.toFixed(4)}
          </div>
        </div>
      </div>
      <CanvasChart
        data={[
          {
            x: data.xs,
            y: data.ys,
            type: 'scatter',
            mode: 'markers',
            marker: { color: colors, size: 7, opacity: 0.7 },
            name: 'Data',
          },
          // Threshold line
          ...(splitFeature === 0
            ? [{
                x: [threshold, threshold],
                y: [0, 1],
                type: 'scatter' as const,
                mode: 'lines' as const,
                line: { color: '#fbbf24', width: 3 },
                name: 'Split',
              }]
            : [{
                x: [0, 1] as number[],
                y: [threshold, threshold],
                type: 'scatter' as const,
                mode: 'lines' as const,
                line: { color: '#fbbf24', width: 3 },
                name: 'Split',
              }]),
        ]}
        layout={{
          xaxis: { title: { text: 'x\u2081' }, range: [-0.1, 1.1] },
          yaxis: { title: { text: 'x\u2082' }, range: [-0.1, 1.1] },
          margin: { t: 20, r: 20, b: 45, l: 55 },
          showlegend: false,
        }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

function xorData(n: number, noise: number): { x1: number[]; x2: number[]; label: number[] } {
  const x1: number[] = [];
  const x2: number[] = [];
  const label: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = pseudoRandom(i * 1.7 + 5) > 0.5 ? 1 : 0;
    const b = pseudoRandom(i * 2.1 + 11) > 0.5 ? 1 : 0;
    const y = a === b ? 0 : 1;
    x1.push(a + gaussianNoise(i * 3.1 + 7, noise));
    x2.push(b + gaussianNoise(i * 4.3 + 13, noise));
    label.push(y);
  }
  return { x1, x2, label };
}

export function TreeEnsembleXorDemo(): React.ReactElement {
  const [n, setN] = useState(400);
  const [noise, setNoise] = useState(0.2);
  const [threshold, setThreshold] = useState(0.6);
  const data = useMemo(() => xorData(n, noise), [n, noise]);
  const probs = useMemo(() => data.x1.map((a, i) => 0.5 + 0.45 * Math.sin(3 * (a - 0.5) * (data.x2[i] - 0.5))), [data]);
  const pred = probs.map((p) => (p > threshold ? 1 : 0));

  // Map numeric label arrays to color strings using Portland colorscale approximation
  const groundTruthColors = data.label.map((l) => (l === 0 ? '#3b82f6' : '#ef4444'));
  const predColors = pred.map((l) => (l === 0 ? '#3b82f6' : '#ef4444'));

  return card(
    'XOR Classification Intuition',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Samples: {n}</label>
          <Slider min={100} max={1000} step={50} value={[n]} onValueChange={([v]) => setN(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Noise: {noise.toFixed(2)}</label>
          <Slider min={0.05} max={0.45} step={0.01} value={[noise]} onValueChange={([v]) => setNoise(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Decision threshold: {threshold.toFixed(2)}</label>
          <Slider min={0.3} max={0.8} step={0.01} value={[threshold]} onValueChange={([v]) => setThreshold(v)} />
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Ground truth</p>
          <CanvasChart
            data={[
              {
                x: data.x1,
                y: data.x2,
                mode: 'markers',
                type: 'scatter',
                name: 'Ground truth',
                marker: { color: groundTruthColors, size: 6, opacity: 0.55 },
              },
            ]}
            layout={{
              margin: { t: 20, r: 20, b: 40, l: 45 },
              showlegend: false,
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Predicted</p>
          <CanvasChart
            data={[
              {
                x: data.x1,
                y: data.x2,
                mode: 'markers',
                type: 'scatter',
                name: 'Predicted',
                marker: { color: predColors, size: 6, opacity: 0.55 },
              },
            ]}
            layout={{
              margin: { t: 20, r: 20, b: 40, l: 45 },
              showlegend: false,
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
      </div>
    </>
  );
}

export function PcaCorrelatedDataDemo(): React.ReactElement {
  const [corr, setCorr] = useState(0.8);
  const [n, setN] = useState(250);
  const data = useMemo(() => {
    const x = Array.from({ length: n }, (_, i) => gaussianNoise(i * 2.7 + 19, 1));
    const y = x.map((xi, i) => corr * xi + gaussianNoise(i * 2.9 + 23, Math.sqrt(Math.max(1 - corr * corr, 0.05))));
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;
    const xc = x.map((v) => v - meanX);
    const yc = y.map((v) => v - meanY);
    const sxx = xc.reduce((a, b) => a + b * b, 0) / (n - 1);
    const syy = yc.reduce((a, b) => a + b * b, 0) / (n - 1);
    const sxy = xc.reduce((a, b, i) => a + b * yc[i], 0) / (n - 1);
    const tr = sxx + syy;
    const det = sxx * syy - sxy * sxy;
    const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
    const lambda1 = tr / 2 + disc;
    const lambda2 = tr / 2 - disc;
    const v1 = Math.abs(sxy) > 1e-8 ? [lambda1 - syy, sxy] : [1, 0];
    const norm = Math.hypot(v1[0], v1[1]) || 1;
    return { x, y, pcx: v1[0] / norm, pcy: v1[1] / norm, lambda1, lambda2 };
  }, [corr, n]);

  const evr1 = (data.lambda1 / (data.lambda1 + data.lambda2)).toFixed(2);

  return card(
    'PCA on Correlated Data',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Correlation: {corr.toFixed(2)}</label>
          <Slider min={0.1} max={0.98} step={0.01} value={[corr]} onValueChange={([v]) => setCorr(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Samples: {n}</label>
          <Slider min={100} max={600} step={20} value={[n]} onValueChange={([v]) => setN(v)} />
        </div>
      </div>
      <div className="mb-2 text-right text-sm text-[var(--text-muted)]">EVR1: {evr1}</div>
      <CanvasChart
        data={[
          { x: data.x, y: data.y, type: 'scatter', mode: 'markers', marker: { color: '#3b82f6', size: 6, opacity: 0.6 }, name: 'Data' },
          { x: [0, 2.5 * data.pcx], y: [0, 2.5 * data.pcy], type: 'scatter', mode: 'lines', line: { color: '#ec4899', width: 4 }, name: 'PC1' },
          { x: [0, -2.5 * data.pcy], y: [0, 2.5 * data.pcx], type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 4 }, name: 'PC2' },
        ]}
        layout={{
          xaxis: { title: { text: 'x1' } },
          yaxis: { title: { text: 'x2' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function ExplainedVarianceDemo(): React.ReactElement {
  const [dims, setDims] = useState(8);
  const eig = useMemo(() => {
    const values = Array.from({ length: dims }, (_, i) => Math.exp(-i / 2.1));
    const total = values.reduce((a, b) => a + b, 0);
    let run = 0;
    const cumulative = values.map((v) => {
      run += v;
      return run / total;
    });
    return { values: values.map((v) => v / total), cumulative };
  }, [dims]);

  const xAxis = Array.from({ length: dims }, (_, i) => i + 1);

  return card(
    'Explained Variance (Scree)',
    <>
      <div className="mb-4">
        <label className="text-sm text-[var(--text-muted)]">Dimensions: {dims}</label>
        <Slider min={3} max={20} step={1} value={[dims]} onValueChange={([v]) => setDims(v)} />
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Per-component variance</p>
          <CanvasChart
            data={[
              { x: xAxis, y: eig.values, type: 'bar', name: 'Per-component' },
            ]}
            layout={{
              xaxis: { title: { text: 'Component' } },
              yaxis: { title: { text: 'Explained variance ratio' } },
              margin: { t: 20, r: 20, b: 45, l: 55 },
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">Cumulative variance</p>
          <CanvasChart
            data={[
              { x: xAxis, y: eig.cumulative, type: 'scatter', mode: 'lines+markers', name: 'Cumulative' },
            ]}
            layout={{
              xaxis: { title: { text: 'Component' } },
              yaxis: { title: { text: 'Cumulative variance' }, range: [0, 1.05] },
              margin: { t: 20, r: 20, b: 45, l: 55 },
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
      </div>
    </>
  );
}

export function TsneUmapComparisonDemo(): React.ReactElement {
  const [clusters, setClusters] = useState(3);
  const [spread, setSpread] = useState(0.5);
  const points = useMemo(() => {
    const centers = Array.from({ length: clusters }, (_, i) => {
      const angle = (2 * Math.PI * i) / clusters;
      return { x: 2 * Math.cos(angle), y: 2 * Math.sin(angle) };
    });
    const xs: number[] = [];
    const ys: number[] = [];
    const labels: number[] = [];
    for (let c = 0; c < centers.length; c++) {
      for (let i = 0; i < 120; i++) {
        const seedBase = c * 1000 + i * 7;
        xs.push(centers[c].x + gaussianNoise(seedBase + 1, spread));
        ys.push(centers[c].y + gaussianNoise(seedBase + 2, spread));
        labels.push(c);
      }
    }
    const pcaX = xs.map((v, i) => 0.8 * v + 0.2 * ys[i]);
    const pcaY = ys.map((v, i) => -0.2 * xs[i] + 0.8 * v);
    const tsneX = xs.map((v, i) => Math.tanh(0.7 * v) + 0.25 * Math.sin(ys[i]));
    const tsneY = ys.map((v, i) => Math.tanh(0.7 * v) + 0.25 * Math.cos(xs[i]));
    const umapX = xs.map((v, i) => 0.9 * Math.tanh(v) + 0.1 * ys[i]);
    const umapY = ys.map((v, i) => 0.9 * Math.tanh(v) - 0.1 * xs[i]);

    // Map cluster index to colors
    const clusterColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
    const pointColors = labels.map((l) => clusterColors[l % clusterColors.length]);
    return { pointColors, pcaX, pcaY, tsneX, tsneY, umapX, umapY };
  }, [clusters, spread]);

  const commonLayout = {
    margin: { t: 20, r: 20, b: 40, l: 40 },
    showlegend: false,
  };

  return card(
    'PCA vs t-SNE vs UMAP (Intuition)',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Cluster count: {clusters}</label>
          <Slider min={2} max={6} step={1} value={[clusters]} onValueChange={([v]) => setClusters(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Spread: {spread.toFixed(2)}</label>
          <Slider min={0.15} max={1.2} step={0.05} value={[spread]} onValueChange={([v]) => setSpread(v)} />
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">PCA</p>
          <CanvasChart
            data={[
              {
                x: points.pcaX,
                y: points.pcaY,
                type: 'scatter',
                mode: 'markers',
                marker: { size: 5, color: points.pointColors },
                name: 'PCA',
              },
            ]}
            layout={commonLayout}
            style={{ width: '100%', height: 280 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">t-SNE</p>
          <CanvasChart
            data={[
              {
                x: points.tsneX,
                y: points.tsneY,
                type: 'scatter',
                mode: 'markers',
                marker: { size: 5, color: points.pointColors },
                name: 't-SNE',
              },
            ]}
            layout={commonLayout}
            style={{ width: '100%', height: 280 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">UMAP</p>
          <CanvasChart
            data={[
              {
                x: points.umapX,
                y: points.umapY,
                type: 'scatter',
                mode: 'markers',
                marker: { size: 5, color: points.pointColors },
                name: 'UMAP',
              },
            ]}
            layout={commonLayout}
            style={{ width: '100%', height: 280 }}
          />
        </div>
      </div>
    </>
  );
}

function ringAdjacency(n: number, radius: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      const d = Math.min((i - j + n) % n, (j - i + n) % n);
      return d <= radius ? 1 : 0;
    })
  );
}

export function GraphConvolutionIntuitionDemo(): React.ReactElement {
  const [n, setN] = useState(12);
  const [radius, setRadius] = useState(1);
  const z = useMemo(() => ringAdjacency(n, radius), [n, radius]);
  return card(
    'Normalized Adjacency Intuition',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Nodes: {n}</label>
          <Slider min={6} max={24} step={1} value={[n]} onValueChange={([v]) => setN(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Neighborhood radius: {radius}</label>
          <Slider min={1} max={4} step={1} value={[radius]} onValueChange={([v]) => setRadius(v)} />
        </div>
      </div>
      <CanvasHeatmap
        data={[{ z, colorscale: 'Magma', showscale: false }]}
        layout={{
          xaxis: { title: { text: 'Node index' } },
          yaxis: { title: { text: 'Node index' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function GraphAdjacencyDemo(): React.ReactElement {
  const [n, setN] = useState(10);
  const [p, setP] = useState(0.25);
  const mat = useMemo(() => {
    const m = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        if (i === j) {
          m[i][j] = 1;
        } else if (pseudoRandom(i * 97 + j * 57 + n * 11 + p * 1000) < p) {
          m[i][j] = 1;
          m[j][i] = 1;
        }
      }
    }
    return m;
  }, [n, p]);

  return card(
    'Random Graph Adjacency',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Nodes: {n}</label>
          <Slider min={6} max={25} step={1} value={[n]} onValueChange={([v]) => setN(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Edge probability: {p.toFixed(2)}</label>
          <Slider min={0.05} max={0.7} step={0.01} value={[p]} onValueChange={([v]) => setP(v)} />
        </div>
      </div>
      <CanvasHeatmap
        data={[{ z: mat, colorscale: 'Viridis', showscale: false }]}
        layout={{
          xaxis: { title: { text: 'Node' } },
          yaxis: { title: { text: 'Node' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function GraphMessagePassingDemo(): React.ReactElement {
  const [n, setN] = useState(12);
  const [steps, setSteps] = useState(2);
  const [source, setSource] = useState(0);
  const series = useMemo(() => {
    const adj = ringAdjacency(n, 1);
    let signal: number[] = Array.from({ length: n }, (_, i) => (i === source ? 1 : 0));
    for (let s = 0; s < steps; s++) {
      const next = Array.from({ length: n }, () => 0);
      for (let i = 0; i < n; i++) {
        let acc = 0;
        let deg = 0;
        for (let j = 0; j < n; j++) {
          if (adj[i][j] > 0) {
            acc += signal[j];
            deg += 1;
          }
        }
        next[i] = deg > 0 ? acc / deg : signal[i];
      }
      signal = next;
    }
    return signal;
  }, [n, steps, source]);

  return card(
    'Message Passing on a Ring Graph',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Nodes: {n}</label>
          <Slider min={6} max={24} step={1} value={[n]} onValueChange={([v]) => setN(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Steps: {steps}</label>
          <Slider min={1} max={8} step={1} value={[steps]} onValueChange={([v]) => setSteps(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Source node: {source}</label>
          <Slider min={0} max={Math.max(0, n - 1)} step={1} value={[source]} onValueChange={([v]) => setSource(v)} />
        </div>
      </div>
      <CanvasChart
        data={[{ x: Array.from({ length: n }, (_, i) => i), y: series, type: 'bar', marker: { color: '#3b82f6' } }]}
        layout={{
          xaxis: { title: { text: 'Node index' } },
          yaxis: { title: { text: 'Signal after aggregation' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}
