'use client';

import dynamic from 'next/dynamic';

const GershgorinCircles = dynamic(
  () => import('@/components/visualization/EigenvalueSimulations').then((mod) => mod.GershgorinCircles),
  { ssr: false }
);

export default function GershgorinPage() {
  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-[var(--text-strong)]">Gershgorin Circles</h1>
      <p className="text-lg text-[var(--text-muted)] mb-8">Each disk contains at least one eigenvalue. Adjust the matrix to see bounds change.</p>
      <GershgorinCircles />
    </main>
  );
}
