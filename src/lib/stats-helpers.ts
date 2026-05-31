export function sum(a: number[]): number {
  return a.reduce((acc, v) => acc + v, 0);
}

export function mean(a: number[]): number {
  return sum(a) / a.length;
}

// Sample variance (divide by n-1) — matches mathjs default (unbiased).
export function variance(a: number[]): number {
  const m = mean(a);
  return a.reduce((acc, v) => acc + (v - m) ** 2, 0) / (a.length - 1);
}

export function randUniform(lo: number, hi: number): number {
  return lo + (hi - lo) * Math.random();
}
