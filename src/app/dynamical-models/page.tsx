import { FEM1D } from '@/components/visualization/FEM1D';

export default function DynamicalModelsPage() {
  return (
    <main className="container mx-auto p-8 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-white">Dynamical Models</h1>
      <p className="text-lg text-gray-300 mb-8">Interactive simulations of dynamical systems and finite element methods.</p>
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4 text-white">FEM 1D Bar Simulation</h2>
        <FEM1D />
      </section>
    </main>
  );
}