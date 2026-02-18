'use client';

import React, { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

const Plotly = dynamic(() => import('react-plotly.js'), { ssr: false });

function linspace(start: number, stop: number, n: number): number[] {
  if (n <= 1) return [start];
  const step = (stop - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + i * step);
}

function pseudoRandom(seed: number): number {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

function gaussianNoise(seed: number, scale: number): number {
  const u1 = Math.max(pseudoRandom(seed), 1e-8);
  const u2 = pseudoRandom(seed + 1.2345);
  const r = Math.sqrt(-2 * Math.log(u1));
  return scale * r * Math.cos(2 * Math.PI * u2);
}

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
  const { mergeLayout } = usePlotlyTheme();
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
      <Plotly
        data={[
          { x, y: zeroOne, type: 'scatter', mode: 'lines', name: 'Zero-One' },
          { x, y: hinge, type: 'scatter', mode: 'lines', name: 'Hinge' },
          { x, y: bce, type: 'scatter', mode: 'lines', name: 'BCE (y=1)' },
          { x, y: mse, type: 'scatter', mode: 'lines', name: 'MSE' },
        ]}
        layout={mergeLayout({
          xaxis: { title: { text: 'Score/Residual' } },
          yaxis: { title: { text: 'Loss' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function LossLandscapeDemo(): React.ReactElement {
  const [noise, setNoise] = useState(0.2);
  const [ripple, setRipple] = useState(2.5);
  const { mergeLayout } = usePlotlyTheme();
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
      <Plotly
        data={[
          {
            x: grid.x,
            y: grid.y,
            z: grid.z,
            type: 'surface',
            colorscale: 'Viridis',
            showscale: false,
          },
        ]}
        layout={mergeLayout({
          scene: {
            xaxis: { title: { text: 'w1' } },
            yaxis: { title: { text: 'w2' } },
            zaxis: { title: { text: 'Loss' } },
          },
          margin: { t: 20, r: 0, b: 0, l: 0 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 420 }}
      />
    </>
  );
}

export function ValidationSplitDemo(): React.ReactElement {
  const [epochs, setEpochs] = useState(40);
  const [overfitStart, setOverfitStart] = useState(20);
  const { mergeLayout } = usePlotlyTheme();
  const curves = useMemo(() => {
    const x = Array.from({ length: epochs }, (_, i) => i + 1);
    const train = x.map((e) => Math.exp(-e / 11) + 0.03 * Math.sin(e / 2));
    const val = x.map((e) => {
      const base = Math.exp(-e / 10);
      const penalty = e > overfitStart ? 0.004 * (e - overfitStart) * (e - overfitStart) : 0;
      return base + 0.07 + penalty;
    });
    return { x, train, val };
  }, [epochs, overfitStart]);

  return card(
    'Train/Validation Split and Early Stopping',
    <>
      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Epochs: {epochs}</label>
          <Slider
            min={15}
            max={80}
            step={1}
            value={[epochs]}
            onValueChange={([v]) => setEpochs(v)}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Overfitting onset: {overfitStart}</label>
          <Slider
            min={5}
            max={60}
            step={1}
            value={[overfitStart]}
            onValueChange={([v]) => setOverfitStart(v)}
          />
        </div>
      </div>
      <Plotly
        data={[
          { x: curves.x, y: curves.train, type: 'scatter', mode: 'lines', name: 'Training loss' },
          { x: curves.x, y: curves.val, type: 'scatter', mode: 'lines', name: 'Validation loss' },
        ]}
        layout={mergeLayout({
          xaxis: { title: { text: 'Epoch' } },
          yaxis: { title: { text: 'Loss' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function TreeSplitImpurityDemo(): React.ReactElement {
  const [p, setP] = useState(0.5);
  const { mergeLayout } = usePlotlyTheme();
  const gini = 1 - (p * p + (1 - p) * (1 - p));
  const entropy = -(p * Math.log2(Math.max(p, 1e-9)) + (1 - p) * Math.log2(Math.max(1 - p, 1e-9)));

  return card(
    'Impurity Measures',
    <>
      <div className="mb-4">
        <label className="text-sm text-[var(--text-muted)]">Class-1 probability: {p.toFixed(2)}</label>
        <Slider
          min={0.01}
          max={0.99}
          step={0.01}
          value={[p]}
          onValueChange={([v]) => setP(v)}
        />
      </div>
      <div className="mb-4 text-sm text-[var(--text-muted)]">
        Gini: {gini.toFixed(4)} | Entropy: {entropy.toFixed(4)}
      </div>
      <Plotly
        data={[
          {
            x: linspace(0.01, 0.99, 100),
            y: linspace(0.01, 0.99, 100).map((t) => 1 - (t * t + (1 - t) * (1 - t))),
            type: 'scatter',
            mode: 'lines',
            name: 'Gini',
          },
          {
            x: linspace(0.01, 0.99, 100),
            y: linspace(0.01, 0.99, 100).map((t) => -(t * Math.log2(t) + (1 - t) * Math.log2(1 - t))),
            type: 'scatter',
            mode: 'lines',
            name: 'Entropy',
          },
          { x: [p], y: [gini], type: 'scatter', mode: 'markers', name: 'Current Gini', marker: { size: 10 } },
          { x: [p], y: [entropy], type: 'scatter', mode: 'markers', name: 'Current Entropy', marker: { size: 10 } },
        ]}
        layout={mergeLayout({
          xaxis: { title: { text: 'p(class 1)' } },
          yaxis: { title: { text: 'Impurity' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        })}
        config={{ displayModeBar: false }}
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
  const { mergeLayout } = usePlotlyTheme();
  const data = useMemo(() => xorData(n, noise), [n, noise]);
  const probs = useMemo(() => data.x1.map((a, i) => 0.5 + 0.45 * Math.sin(3 * (a - 0.5) * (data.x2[i] - 0.5))), [data]);
  const pred = probs.map((p) => (p > threshold ? 1 : 0));

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
      <Plotly
        data={[
          {
            x: data.x1,
            y: data.x2,
            mode: 'markers',
            type: 'scatter',
            name: 'Ground truth',
            marker: { color: data.label, colorscale: 'Viridis', size: 6, opacity: 0.55 },
            xaxis: 'x',
            yaxis: 'y',
          },
          {
            x: data.x1,
            y: data.x2,
            mode: 'markers',
            type: 'scatter',
            name: 'Predicted',
            marker: { color: pred, colorscale: 'Portland', size: 6, opacity: 0.55 },
            xaxis: 'x2',
            yaxis: 'y2',
          },
        ]}
        layout={mergeLayout({
          grid: { rows: 1, columns: 2, pattern: 'independent' },
          margin: { t: 20, r: 20, b: 40, l: 45 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function PcaCorrelatedDataDemo(): React.ReactElement {
  const [corr, setCorr] = useState(0.8);
  const [n, setN] = useState(250);
  const { mergeLayout } = usePlotlyTheme();
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
      <Plotly
        data={[
          { x: data.x, y: data.y, type: 'scatter', mode: 'markers', marker: { color: '#3b82f6', size: 6, opacity: 0.6 }, name: 'Data' },
          { x: [0, 2.5 * data.pcx], y: [0, 2.5 * data.pcy], type: 'scatter', mode: 'lines', line: { color: '#ec4899', width: 4 }, name: 'PC1' },
          { x: [0, -2.5 * data.pcy], y: [0, 2.5 * data.pcx], type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 4 }, name: 'PC2' },
        ]}
        layout={mergeLayout({
          xaxis: { title: { text: 'x1' } },
          yaxis: { title: { text: 'x2' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
          annotations: [
            {
              xref: 'paper',
              yref: 'paper',
              x: 1,
              y: 1.1,
              text: `EVR1 ${(data.lambda1 / (data.lambda1 + data.lambda2)).toFixed(2)}`,
              showarrow: false,
            },
          ],
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function ExplainedVarianceDemo(): React.ReactElement {
  const [dims, setDims] = useState(8);
  const { mergeLayout } = usePlotlyTheme();
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

  return card(
    'Explained Variance (Scree)',
    <>
      <div className="mb-4">
        <label className="text-sm text-[var(--text-muted)]">Dimensions: {dims}</label>
        <Slider min={3} max={20} step={1} value={[dims]} onValueChange={([v]) => setDims(v)} />
      </div>
      <Plotly
        data={[
          { x: Array.from({ length: dims }, (_, i) => i + 1), y: eig.values, type: 'bar', name: 'Per-component' },
          { x: Array.from({ length: dims }, (_, i) => i + 1), y: eig.cumulative, type: 'scatter', mode: 'lines+markers', name: 'Cumulative', yaxis: 'y2' },
        ]}
        layout={mergeLayout({
          xaxis: { title: { text: 'Component' } },
          yaxis: { title: { text: 'Explained variance ratio' } },
          yaxis2: { title: { text: 'Cumulative' }, overlaying: 'y', side: 'right', range: [0, 1.05] },
          margin: { t: 20, r: 45, b: 45, l: 55 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function TsneUmapComparisonDemo(): React.ReactElement {
  const [clusters, setClusters] = useState(3);
  const [spread, setSpread] = useState(0.5);
  const { mergeLayout } = usePlotlyTheme();
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
    return { labels, pcaX, pcaY, tsneX, tsneY, umapX, umapY };
  }, [clusters, spread]);

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
      <Plotly
        data={[
          { x: points.pcaX, y: points.pcaY, type: 'scatter', mode: 'markers', marker: { size: 5, color: points.labels, colorscale: 'Viridis' }, name: 'PCA', xaxis: 'x', yaxis: 'y' },
          { x: points.tsneX, y: points.tsneY, type: 'scatter', mode: 'markers', marker: { size: 5, color: points.labels, colorscale: 'Viridis' }, name: 't-SNE', xaxis: 'x2', yaxis: 'y2' },
          { x: points.umapX, y: points.umapY, type: 'scatter', mode: 'markers', marker: { size: 5, color: points.labels, colorscale: 'Viridis' }, name: 'UMAP', xaxis: 'x3', yaxis: 'y3' },
        ]}
        layout={mergeLayout({
          grid: { rows: 1, columns: 3, pattern: 'independent' },
          margin: { t: 20, r: 20, b: 40, l: 40 },
          showlegend: false,
          annotations: [
            { x: 0.13, y: 1.08, xref: 'paper', yref: 'paper', text: 'PCA', showarrow: false },
            { x: 0.50, y: 1.08, xref: 'paper', yref: 'paper', text: 't-SNE', showarrow: false },
            { x: 0.87, y: 1.08, xref: 'paper', yref: 'paper', text: 'UMAP', showarrow: false },
          ],
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
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
  const { mergeLayout } = usePlotlyTheme();
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
      <Plotly
        data={[{ z, type: 'heatmap', colorscale: 'Magma', showscale: false }]}
        layout={mergeLayout({
          xaxis: { title: { text: 'Node index' } },
          yaxis: { title: { text: 'Node index' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function GraphAdjacencyDemo(): React.ReactElement {
  const [n, setN] = useState(10);
  const [p, setP] = useState(0.25);
  const { mergeLayout } = usePlotlyTheme();
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
      <Plotly
        data={[{ z: mat, type: 'heatmap', colorscale: 'Viridis', showscale: false }]}
        layout={mergeLayout({
          xaxis: { title: { text: 'Node' } },
          yaxis: { title: { text: 'Node' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}

export function GraphMessagePassingDemo(): React.ReactElement {
  const [n, setN] = useState(12);
  const [steps, setSteps] = useState(2);
  const [source, setSource] = useState(0);
  const { mergeLayout } = usePlotlyTheme();
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
      <Plotly
        data={[{ x: Array.from({ length: n }, (_, i) => i), y: series, type: 'bar', marker: { color: '#3b82f6' } }]}
        layout={mergeLayout({
          xaxis: { title: { text: 'Node index' } },
          yaxis: { title: { text: 'Signal after aggregation' } },
          margin: { t: 20, r: 20, b: 45, l: 55 },
        })}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: 360 }}
      />
    </>
  );
}
