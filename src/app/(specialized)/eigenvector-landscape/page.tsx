'use client';

import dynamic from 'next/dynamic';

const EigenvectorLandscape = dynamic(
  () => import('@/components/visualization/EigenvectorLandscape').then((mod) => mod.EigenvectorLandscape),
  { ssr: false }
);

export default function EigenvectorLandscapePage() {
  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-[var(--text-strong)]">Eigenvector Landscape (Quadratic Form + Ball Roll)</h1>
      <p className="text-lg text-[var(--text-muted)] mb-8">Interactive 3D quadratic form surface. Watch the ball roll downhill along principal directions.</p>
      <EigenvectorLandscape />
    </main>
  );
}
