import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { BlockMath } from "@/components/math";

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-4xl px-6 py-12">
        <h1 className="text-4xl font-bold text-gray-900">About Koala-Brain</h1>
        <p className="mt-4 text-lg text-gray-600">
          Interactive computational physics education — where precision meets accessibility.
        </p>

        <div className="mt-12 space-y-8">
          <Card>
            <CardHeader>
              <CardTitle>Our Mission</CardTitle>
            </CardHeader>
            <CardContent className="text-gray-600">
              <p>
                Koala-Brain aims to be the best interactive computational physics learning resource 
                on the web. We believe that complex physics concepts become intuitive when paired 
                with interactive visualizations and rigorous mathematical treatment.
              </p>
              <p className="mt-4">
                Our content is designed for masters-level students and curious minds who want to 
                understand the underlying mathematics while seeing concepts come to life through 
                simulations and visualizations.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>What Makes Us Different</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 sm:grid-cols-2">
                <div>
                  <h3 className="font-semibold text-gray-900">📐 Mathematical Rigor</h3>
                  <p className="mt-2 text-sm text-gray-600">
                    Every concept is backed by proper mathematical foundations with LaTeX-rendered 
                    equations like the Schrödinger equation:
                  </p>
                  <div className="mt-3">
                    <BlockMath>{`i\\hbar\\frac{\\partial}{\\partial t}|\\psi\\rangle = \\hat{H}|\\psi\\rangle`}</BlockMath>
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">📊 Interactive Visualizations</h3>
                  <p className="mt-2 text-sm text-gray-600">
                    Explore physics through hands-on simulations. Adjust parameters in real-time 
                    and see how systems evolve.
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">🎯 Structured Learning Paths</h3>
                  <p className="mt-2 text-sm text-gray-600">
                    Content organized by difficulty level, from beginner to expert. Clear 
                    prerequisites and learning objectives for each topic.
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">💻 Code Examples</h3>
                  <p className="mt-2 text-sm text-gray-600">
                    Practical implementations in Python and TypeScript. Learn how to implement 
                    the algorithms yourself.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Topics Covered</CardTitle>
            </CardHeader>
            <CardContent className="text-gray-600">
              <ul className="space-y-2">
                <li><strong>Quantum Optics</strong> — Wigner functions, coherent states, quantum measurement</li>
                <li><strong>Continuum Mechanics</strong> — Rod dynamics, iceberg simulations, gravity waves</li>
                <li><strong>Inverse Problems</strong> — Tikhonov regularization, practical inversion</li>
                <li><strong>Complex Physics</strong> — Statistical mechanics, percolation, fractals</li>
                <li><strong>Scientific Computing</strong> — Numerical methods, optimization</li>
                <li><strong>Deep Learning</strong> — GANs, VAEs, CNNs, U-Net</li>
                <li><strong>Online Learning</strong> — Multi-armed bandits, regret analysis</li>
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Technology Stack</CardTitle>
            </CardHeader>
            <CardContent className="text-gray-600">
              <p>
                Built with modern web technologies for the best learning experience:
              </p>
              <ul className="mt-4 space-y-1">
                <li>• <strong>Next.js 16</strong> — React framework with App Router</li>
                <li>• <strong>TypeScript</strong> — Type-safe code and examples</li>
                <li>• <strong>KaTeX</strong> — Fast LaTeX rendering</li>
                <li>• <strong>Plotly / D3 / Three.js</strong> — Interactive visualizations</li>
                <li>• <strong>Tailwind CSS</strong> — Beautiful, responsive design</li>
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Open Source</CardTitle>
            </CardHeader>
            <CardContent className="text-gray-600">
              <p>
                Koala-Brain is open source. Contributions are welcome! Find us on GitHub:
              </p>
              <a 
                href="https://github.com/tonton-golio/computational_physics"
                target="_blank"
                rel="noopener noreferrer"
                className="mt-4 inline-block text-primary-600 hover:underline"
              >
                github.com/tonton-golio/computational_physics
              </a>
              <p className="mt-4 text-sm text-gray-500">
                <a href="/graph" className="text-primary-600 hover:underline">← Back to Explore</a>
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
