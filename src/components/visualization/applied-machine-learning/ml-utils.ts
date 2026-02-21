/**
 * Shared utilities for Applied Machine Learning simulations.
 */

export function linspace(start: number, stop: number, n: number): number[] {
  if (n <= 1) return [start];
  const step = (stop - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + i * step);
}

/** Deterministic pseudo-random in [0,1). */
export function pseudoRandom(seed: number): number {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

/** Mulberry32 – a full-period 32-bit PRNG. Returns a callable that yields values in [0,1). */
export function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Gaussian noise using Box-Muller transform with deterministic seed. */
export function gaussianNoise(seed: number, scale: number): number {
  const u1 = Math.max(pseudoRandom(seed), 1e-8);
  const u2 = pseudoRandom(seed + 1.2345);
  const r = Math.sqrt(-2 * Math.log(u1));
  return scale * r * Math.cos(2 * Math.PI * u2);
}

/** Gaussian pair from a PRNG function (Box-Muller). */
export function gaussianPair(rng: () => number): [number, number] {
  const u1 = Math.max(rng(), 1e-8);
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

/** Cluster color palette. */
export const CLUSTER_COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899',
  '#06b6d4', '#84cc16', '#f97316', '#6366f1',
] as const;

/**
 * Minimal t-SNE implementation (O(n^2) – suitable for up to ~2000 points).
 * Returns 2D coordinates for each input point.
 */
export function tsne(
  data: number[][],
  opts: { perplexity?: number; iterations?: number; learningRate?: number; seed?: number } = {},
): number[][] {
  const { perplexity = 30, iterations = 300, learningRate = 100, seed = 42 } = opts;
  const n = data.length;
  const dim = data[0].length;
  const rng = mulberry32(seed);

  // Pairwise squared distances in high-dimensional space
  const dist2 = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let d = 0;
      for (let k = 0; k < dim; k++) {
        const diff = data[i][k] - data[j][k];
        d += diff * diff;
      }
      dist2[i * n + j] = d;
      dist2[j * n + i] = d;
    }
  }

  // Compute conditional probabilities p_{j|i} using binary search for sigma
  const P = new Float64Array(n * n);
  const targetH = Math.log(perplexity);

  for (let i = 0; i < n; i++) {
    let lo = 1e-10;
    let hi = 1e4;
    let beta = 1.0; // 1 / (2 * sigma^2)

    for (let iter = 0; iter < 50; iter++) {
      let sumP = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        P[i * n + j] = Math.exp(-dist2[i * n + j] * beta);
        sumP += P[i * n + j];
      }
      sumP = Math.max(sumP, 1e-10);
      let H = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        P[i * n + j] /= sumP;
        if (P[i * n + j] > 1e-10) {
          H -= P[i * n + j] * Math.log(P[i * n + j]);
        }
      }
      if (Math.abs(H - targetH) < 1e-5) break;
      if (H > targetH) {
        lo = beta;
        beta = hi === 1e4 ? beta * 2 : (beta + hi) / 2;
      } else {
        hi = beta;
        beta = (beta + lo) / 2;
      }
    }
  }

  // Symmetrize: P_ij = (p_{j|i} + p_{i|j}) / (2n)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const pij = (P[i * n + j] + P[j * n + i]) / (2 * n);
      P[i * n + j] = pij;
      P[j * n + i] = pij;
    }
  }

  // Initialize 2D embedding randomly
  const Y = Array.from({ length: n }, () => {
    const [a, b] = gaussianPair(rng);
    return [a * 0.01, b * 0.01];
  });

  const gains = Array.from({ length: n }, () => [1, 1]);
  const yVel = Array.from({ length: n }, () => [0, 0]);
  const momentum = 0.5;
  const finalMomentum = 0.8;

  for (let iter = 0; iter < iterations; iter++) {
    const mom = iter < 100 ? momentum : finalMomentum;

    // Compute Q distribution (Student-t with 1 DOF)
    let sumQ = 0;
    const qNum = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = Y[i][0] - Y[j][0];
        const dy = Y[i][1] - Y[j][1];
        const q = 1 / (1 + dx * dx + dy * dy);
        qNum[i * n + j] = q;
        qNum[j * n + i] = q;
        sumQ += 2 * q;
      }
    }
    sumQ = Math.max(sumQ, 1e-10);

    // Gradient
    const grad = Array.from({ length: n }, () => [0, 0]);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const q = qNum[i * n + j] / sumQ;
        const mult = 4 * (P[i * n + j] - q) * qNum[i * n + j];
        grad[i][0] += mult * (Y[i][0] - Y[j][0]);
        grad[i][1] += mult * (Y[i][1] - Y[j][1]);
      }
    }

    // Update with momentum and adaptive gains
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 2; d++) {
        const sign = grad[i][d] > 0 ? 1 : -1;
        const velSign = yVel[i][d] > 0 ? 1 : -1;
        gains[i][d] = sign !== velSign ? gains[i][d] + 0.2 : Math.max(gains[i][d] * 0.8, 0.01);
        yVel[i][d] = mom * yVel[i][d] - learningRate * gains[i][d] * grad[i][d];
        Y[i][d] += yVel[i][d];
      }
    }

    // Re-center
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += Y[i][0]; my += Y[i][1]; }
    mx /= n; my /= n;
    for (let i = 0; i < n; i++) { Y[i][0] -= mx; Y[i][1] -= my; }
  }

  return Y;
}

/**
 * Principal Component Analysis – returns the top-k 2D projection.
 * For simplicity, always projects to 2D.
 */
export function pca2d(data: number[][]): number[][] {
  const n = data.length;
  const dim = data[0].length;

  // Center data
  const mean = new Array(dim).fill(0);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) mean[d] += data[i][d];
  }
  for (let d = 0; d < dim; d++) mean[d] /= n;
  const centered = data.map(row => row.map((v, d) => v - mean[d]));

  // Covariance matrix (dim x dim)
  const cov: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < dim; a++) {
      for (let b = a; b < dim; b++) {
        const v = centered[i][a] * centered[i][b];
        cov[a][b] += v;
        if (a !== b) cov[b][a] += v;
      }
    }
  }
  for (let a = 0; a < dim; a++) {
    for (let b = 0; b < dim; b++) cov[a][b] /= (n - 1);
  }

  // Power iteration for top 2 eigenvectors
  const eigvecs: number[][] = [];
  const covCopy = cov.map(r => [...r]);

  for (let ev = 0; ev < Math.min(2, dim); ev++) {
    let v = new Array(dim).fill(0).map((_, i) => pseudoRandom(i + ev * 100 + 7));
    let norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
    v = v.map(x => x / norm);

    for (let iter = 0; iter < 100; iter++) {
      const newV = new Array(dim).fill(0);
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          newV[i] += covCopy[i][j] * v[j];
        }
      }
      norm = Math.sqrt(newV.reduce((a, b) => a + b * b, 0)) || 1;
      v = newV.map(x => x / norm);
    }
    eigvecs.push(v);

    // Deflate covariance
    const lambda = v.reduce((acc, vi, i) => {
      let s = 0;
      for (let j = 0; j < dim; j++) s += covCopy[i][j] * v[j];
      return acc + vi * s;
    }, 0);
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        covCopy[i][j] -= lambda * v[i] * v[j];
      }
    }
  }

  // Project
  return centered.map(row => {
    return eigvecs.map(ev => ev.reduce((acc, vi, d) => acc + vi * row[d], 0));
  });
}

/**
 * Generate Swiss Roll dataset: 3D points on a rolled manifold.
 * Returns { data: number[][] (n x 3), labels: number[] (continuous position on roll) }
 */
export function generateSwissRoll(
  n: number,
  seed: number = 42,
): { data: number[][]; labels: number[] } {
  const rng = mulberry32(seed);
  const data: number[][] = [];
  const labels: number[] = [];

  for (let i = 0; i < n; i++) {
    const t = 1.5 * Math.PI * (1 + 2 * rng()); // angle
    const height = 21 * rng(); // height along roll
    const x = t * Math.cos(t);
    const y = height;
    const z = t * Math.sin(t);
    data.push([x, y, z]);
    labels.push(t); // color by angle
  }

  return { data, labels };
}
