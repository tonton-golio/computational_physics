import { MandelbrotFractal } from '@/components/visualization/complex-physics/MandelbrotFractal';
import { GameOfLife } from '@/components/visualization/complex-physics/GameOfLife';
import { LorenzAttractor } from '@/components/visualization/complex-physics/LorenzAttractor';

export default function ComplexPhysicsPage() {
  return (
    <main className="container mx-auto p-8 max-w-6xl">
      <h1 className="text-4xl font-bold mb-8 text-white">Complex Physics</h1>
      <p className="text-lg text-gray-300 mb-8">
        Interactive visualizations of complex systems physics. Explore emergent behavior, self-organization,
        phase transitions, and chaotic dynamics through simulations and visualizations.
      </p>

      <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 mb-12">
        <section className="bg-[#151525] p-6 rounded-lg border border-[#2d2d44]">
          <h2 className="text-xl font-semibold mb-3 text-white">Fractals & Chaos</h2>
          <p className="text-gray-300 mb-4 text-sm">
            Self-similar patterns and strange attractors reveal the beauty of deterministic chaos.
          </p>
          <div className="space-y-4">
            <MandelbrotFractal />
            <LorenzAttractor />
          </div>
        </section>

        <section className="bg-[#151525] p-6 rounded-lg border border-[#2d2d44]">
          <h2 className="text-xl font-semibold mb-3 text-white">Cellular Automata</h2>
          <p className="text-gray-300 mb-4 text-sm">
            Simple rules generate complex emergent behavior in discrete systems.
          </p>
          <GameOfLife />
        </section>

        <section className="bg-[#151525] p-6 rounded-lg border border-[#2d2d44]">
          <h2 className="text-xl font-semibold mb-3 text-white">Statistical Mechanics</h2>
          <p className="text-gray-300 mb-4 text-sm">
            Phase transitions and critical phenomena in many-body systems.
          </p>
          <div className="text-center py-8 text-gray-400">
            <p className="text-sm">See detailed simulations in the content sections below</p>
            <p className="text-xs mt-2">Ising model, percolation, networks</p>
          </div>
        </section>
      </div>

      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-white">Explore Topics</h2>
        <p className="text-gray-300 mb-6">
          Dive deeper into specific complex systems concepts using the interactive graph navigation above.
          Each topic includes detailed explanations, mathematical derivations, and interactive simulations.
        </p>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div className="bg-[#151525] p-4 rounded-lg border border-[#2d2d44]">
            <h3 className="text-lg font-medium text-white mb-2">Phase Transitions</h3>
            <p className="text-gray-400 text-sm">Critical phenomena, Ising model, mean-field theory</p>
          </div>
          <div className="bg-[#151525] p-4 rounded-lg border border-[#2d2d44]">
            <h3 className="text-lg font-medium text-white mb-2">Networks</h3>
            <p className="text-gray-400 text-sm">Scale-free networks, centrality, modularity</p>
          </div>
          <div className="bg-[#151525] p-4 rounded-lg border border-[#2d2d44]">
            <h3 className="text-lg font-medium text-white mb-2">Percolation & Fractals</h3>
            <p className="text-gray-400 text-sm">Connectivity, fractal dimension, criticality</p>
          </div>
          <div className="bg-[#151525] p-4 rounded-lg border border-[#2d2d44]">
            <h3 className="text-lg font-medium text-white mb-2">Self-Organization</h3>
            <p className="text-gray-400 text-sm">Emergent patterns, criticality, sandpile model</p>
          </div>
          <div className="bg-[#151525] p-4 rounded-lg border border-[#2d2d44]">
            <h3 className="text-lg font-medium text-white mb-2">Agent-Based Models</h3>
            <p className="text-gray-400 text-sm">Gillespie algorithm, stochastic simulations</p>
          </div>
          <div className="bg-[#151525] p-4 rounded-lg border border-[#2d2d44]">
            <h3 className="text-lg font-medium text-white mb-2">Econophysics</h3>
            <p className="text-gray-400 text-sm">Financial markets, Brownian motion, risk models</p>
          </div>
        </div>
      </section>
    </main>
  );
}