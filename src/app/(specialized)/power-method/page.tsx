'use client';

import dynamic from 'next/dynamic';

const PowerMethodAnimation = dynamic(
  () => import('@/components/visualization/EigenvalueSimulations').then((mod) => mod.PowerMethodAnimation),
  { ssr: false }
);

export default function PowerMethodPage() {
  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-[var(--text-strong)]">Power Method Animation</h1>
      <p className="text-lg text-[var(--text-muted)] mb-8">Watch how repeated matrix-vector multiplication converges to the dominant eigenvector.</p>
      <PowerMethodAnimation />
    </main>
  );
}
