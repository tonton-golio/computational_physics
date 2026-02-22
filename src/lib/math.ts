/**
 * Shared math utilities used across simulation components.
 */

/** Generate `n` evenly spaced values from `start` to `stop` (inclusive). */
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

/** Mulberry32 -- a full-period 32-bit PRNG. Returns a callable that yields values in [0,1). */
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

/** Clamp `value` to [min, max]. */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Gaussian probability density function. */
export function gaussPdf(x: number, mu: number, sigma: number): number {
  return (1 / (sigma * Math.sqrt(2 * Math.PI))) *
    Math.exp(-0.5 * ((x - mu) / sigma) ** 2);
}
