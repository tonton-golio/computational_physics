import { PowerMethodAnimation } from '@/components/visualization/EigenvalueSimulations';

export default function PowerMethodPage() {
  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-white">Power Method Animation</h1>
      <p className="text-lg text-gray-300 mb-8">Watch how repeated matrix-vector multiplication converges to the dominant eigenvector.</p>
      <PowerMethodAnimation />
    </main>
  );
}