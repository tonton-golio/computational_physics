'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PCAVisualizationProps {
  id?: string;
}

// Seeded pseudo-random number generator for reproducibility
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng: () => number): number {
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export default function PCAVisualization({ id }: PCAVisualizationProps) {
  const [nClusters, setNClusters] = useState(3);
  const [nPoints, setNPoints] = useState(80);
  const [spread, setSpread] = useState(0.6);
  const [showPCs, setShowPCs] = useState(true);

  const { points, pc1, pc2, eigenvalues, center } = useMemo(() => {
    const rng = mulberry32(42);

    // Generate cluster centers
    const clusterCenters: [number, number][] = [];
    for (let c = 0; c < nClusters; c++) {
      clusterCenters.push([
        gaussianRandom(rng) * 3,
        gaussianRandom(rng) * 3,
      ]);
    }

    // Generate points around cluster centers
    const pts: { x: number; y: number; cluster: number }[] = [];
    const pointsPerCluster = Math.floor(nPoints / nClusters);
    for (let c = 0; c < nClusters; c++) {
      for (let i = 0; i < pointsPerCluster; i++) {
        pts.push({
          x: clusterCenters[c][0] + gaussianRandom(rng) * spread,
          y: clusterCenters[c][1] + gaussianRandom(rng) * spread,
          cluster: c,
        });
      }
    }

    // Compute mean
    const meanX = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const meanY = pts.reduce((s, p) => s + p.y, 0) / pts.length;

    // Compute covariance matrix
    let cxx = 0,
      cxy = 0,
      cyy = 0;
    for (const p of pts) {
      const dx = p.x - meanX;
      const dy = p.y - meanY;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    }
    const n = pts.length;
    cxx /= n;
    cxy /= n;
    cyy /= n;

    // Eigenvalues of 2x2 symmetric matrix
    const trace = cxx + cyy;
    const det = cxx * cyy - cxy * cxy;
    const discriminant = Math.sqrt(Math.max(0, trace * trace / 4 - det));
    const lambda1 = trace / 2 + discriminant;
    const lambda2 = trace / 2 - discriminant;

    // Eigenvectors
    let v1x: number, v1y: number, v2x: number, v2y: number;
    if (Math.abs(cxy) > 1e-10) {
      v1x = lambda1 - cyy;
      v1y = cxy;
      v2x = lambda2 - cyy;
      v2y = cxy;
    } else {
      v1x = 1;
      v1y = 0;
      v2x = 0;
      v2y = 1;
    }

    // Normalize
    const len1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const len2 = Math.sqrt(v2x * v2x + v2y * v2y);
    v1x /= len1;
    v1y /= len1;
    v2x /= len2;
    v2y /= len2;

    // Scale eigenvectors by sqrt of eigenvalue for visualization
    const scale1 = Math.sqrt(lambda1) * 2;
    const scale2 = Math.sqrt(lambda2) * 2;

    return {
      points: pts,
      pc1: { x: v1x * scale1, y: v1y * scale1 },
      pc2: { x: v2x * scale2, y: v2y * scale2 },
      eigenvalues: [lambda1, lambda2],
      center: { x: meanX, y: meanY },
    };
  }, [nClusters, nPoints, spread]);

  const clusterColors = [
    '#ff6b6b',
    '#4ecdc4',
    '#45b7d1',
    '#f9ca24',
    '#6c5ce7',
  ];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const plotData: any[] = [];

  // Add scatter points for each cluster
  for (let c = 0; c < nClusters; c++) {
    const clusterPts = points.filter((p) => p.cluster === c);
    plotData.push({
      type: 'scatter' as const,
      x: clusterPts.map((p) => p.x),
      y: clusterPts.map((p) => p.y),
      mode: 'markers' as const,
      marker: {
        color: clusterColors[c % clusterColors.length],
        size: 7,
        opacity: 0.8,
      },
      name: `Cluster ${c + 1}`,
    });
  }

  // Add principal component arrows
  if (showPCs) {
    plotData.push({
      type: 'scatter' as const,
      x: [center.x - pc1.x, center.x, center.x + pc1.x],
      y: [center.y - pc1.y, center.y, center.y + pc1.y],
      mode: 'lines' as const,
      line: { color: '#f39c12', width: 3 },
      name: `PC1 (var: ${(eigenvalues[0] / (eigenvalues[0] + eigenvalues[1]) * 100).toFixed(1)}%)`,
    });
    plotData.push({
      type: 'scatter' as const,
      x: [center.x - pc2.x, center.x, center.x + pc2.x],
      y: [center.y - pc2.y, center.y, center.y + pc2.y],
      mode: 'lines' as const,
      line: { color: '#e74c3c', width: 3, dash: 'dash' },
      name: `PC2 (var: ${(eigenvalues[1] / (eigenvalues[0] + eigenvalues[1]) * 100).toFixed(1)}%)`,
    });

    // Center marker
    plotData.push({
      type: 'scatter' as const,
      x: [center.x],
      y: [center.y],
      mode: 'markers' as const,
      marker: { color: '#ffffff', size: 12, symbol: 'x' },
      name: 'Mean',
      showlegend: false,
    });
  }

  const totalVar = eigenvalues[0] + eigenvalues[1];

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">
        Interactive PCA Visualization
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Number of Clusters: {nClusters}
            </label>
            <input
              type="range"
              min={2}
              max={5}
              step={1}
              value={nClusters}
              onChange={(e) => setNClusters(parseInt(e.target.value))}
              className="w-full accent-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Points per Cluster: {Math.floor(nPoints / nClusters)}
            </label>
            <input
              type="range"
              min={30}
              max={200}
              step={10}
              value={nPoints}
              onChange={(e) => setNPoints(parseInt(e.target.value))}
              className="w-full accent-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Cluster Spread: {spread.toFixed(2)}
            </label>
            <input
              type="range"
              min={0.1}
              max={2.0}
              step={0.1}
              value={spread}
              onChange={(e) => setSpread(parseFloat(e.target.value))}
              className="w-full accent-blue-500"
            />
          </div>
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showPCs}
              onChange={(e) => setShowPCs(e.target.checked)}
              className="accent-blue-500"
            />
            Show Principal Components
          </label>

          <div className="mt-4 p-3 bg-[#0a0a15] rounded text-sm text-gray-300">
            <p className="font-semibold text-white mb-2">Explained Variance:</p>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span style={{ color: '#f39c12' }}>PC1:</span>
                <span>
                  {totalVar > 0
                    ? ((eigenvalues[0] / totalVar) * 100).toFixed(1)
                    : 0}
                  %
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: '#e74c3c' }}>PC2:</span>
                <span>
                  {totalVar > 0
                    ? ((eigenvalues[1] / totalVar) * 100).toFixed(1)
                    : 0}
                  %
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Plot */}
        <div className="lg:col-span-2">
          <Plot
            data={plotData}
            layout={{
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#ccc' },
              xaxis: {
                title: { text: 'Feature 1' },
                gridcolor: 'rgba(255,255,255,0.1)',
                zerolinecolor: 'rgba(255,255,255,0.2)',
              },
              yaxis: {
                title: { text: 'Feature 2' },
                gridcolor: 'rgba(255,255,255,0.1)',
                zerolinecolor: 'rgba(255,255,255,0.2)',
                scaleanchor: 'x',
              },
              legend: {
                bgcolor: 'rgba(0,0,0,0.5)',
                font: { color: '#ccc' },
              },
              margin: { t: 30, b: 50, l: 60, r: 30 },
              autosize: true,
            }}
            useResizeHandler
            style={{ width: '100%', height: '450px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>
    </div>
  );
}
