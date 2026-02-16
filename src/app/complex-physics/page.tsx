import { MandelbrotFractal } from '@/components/visualization/complex-physics/MandelbrotFractal';
import { GameOfLife } from '@/components/visualization/complex-physics/GameOfLife';
import { LorenzAttractor } from '@/components/visualization/complex-physics/LorenzAttractor';

export default function ComplexPhysicsPage() {
  return (
    <main className="container mx-auto p-8 max-w-6xl">
      <h1 className="text-4xl font-bold mb-8 text-white">Complex Physics</h1>
      <p className="text-lg text-gray-300 mb-8">
        Interactive visualizations of complex systems: Mandelbrot set, Conway's Game of Life, and Lorenz attractor.
      </p>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4 text-white">Mandelbrot Set Fractal</h2>
        <p className="text-gray-300 mb-4">Zoom and pan to explore the fractal.</p>
        <MandelbrotFractal />
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4 text-white">Conway's Game of Life</h2>
        <p className="text-gray-300 mb-4">Cellular automaton simulation. Start, stop, or reset.</p>
        <GameOfLife />
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-4 text-white">Lorenz Attractor</h2>
        <p className="text-gray-300 mb-4">Strange attractor with adjustable parameters.</p>
        <LorenzAttractor />
      </section>
    </main>
  );
}