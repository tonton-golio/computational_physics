import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

const featuredTopics = [
  {
    title: "Quantum Optics",
    description: "Wigner functions, phase space dynamics, and quantum state visualization",
    href: "/topics/quantum-optics",
    difficulty: "Expert",
    color: "bg-purple-100 text-purple-800",
  },
  {
    title: "Continuum Mechanics",
    description: "Rod dynamics, iceberg simulations, and gravity waves",
    href: "/topics/continuum-mechanics",
    difficulty: "Expert",
    color: "bg-blue-100 text-blue-800",
  },
  {
    title: "Inverse Problems",
    description: "Tikhonov regularization and practical inversion techniques",
    href: "/topics/inverse-problems",
    difficulty: "Advanced",
    color: "bg-green-100 text-green-800",
  },
  {
    title: "Complex Physics",
    description: "Statistical mechanics, percolation, and fractals",
    href: "/topics/complex-physics",
    difficulty: "Intermediate",
    color: "bg-orange-100 text-orange-800",
  },
  {
    title: "Scientific Computing",
    description: "Linear equations, optimization, and numerical methods",
    href: "/topics/scientific-computing",
    difficulty: "Intermediate",
    color: "bg-cyan-100 text-cyan-800",
  },
  {
    title: "Online Reinforcement Learning",
    description: "Multi-armed bandits and regret analysis",
    href: "/topics/online-reinforcement-learning",
    difficulty: "Intermediate",
    color: "bg-pink-100 text-pink-800",
  },
  {
    title: "Advanced Deep Learning",
    description: "GANs, VAEs, CNNs, and U-Net architectures",
    href: "/topics/advanced-deep-learning",
    difficulty: "Advanced",
    color: "bg-red-100 text-red-800",
  },
  {
    title: "Applied Statistics",
    description: "Statistical methods and data analysis techniques",
    href: "/topics/applied-statistics",
    difficulty: "Beginner",
    color: "bg-yellow-100 text-yellow-800",
  },
];

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      {/* Hero Section */}
      <header className="mx-auto max-w-7xl px-6 py-24 sm:py-32">
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
            Koala-Brain
          </h1>
          <p className="mt-6 text-lg leading-8 text-gray-600 sm:text-xl">
            Computational physics education — where precision meets accessibility.
          </p>
          <p className="mt-4 text-base text-gray-500">
            Masters-level content with interactive visualizations
          </p>
          <div className="mt-10 flex items-center justify-center gap-4">
            <Button href="/topics" size="lg">
              Explore Topics
            </Button>
            <Button href="/about" variant="outline" size="lg">
              About
            </Button>
          </div>
        </div>
      </header>

      {/* Topics Grid */}
      <main className="mx-auto max-w-7xl px-6 pb-24">
        <h2 className="mb-8 text-2xl font-semibold text-gray-900">Topics</h2>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {featuredTopics.map((topic) => (
            <Link key={topic.title} href={topic.href}>
              <Card className="h-full transition-shadow hover:shadow-md cursor-pointer">
                <CardHeader>
                  <div className="mb-2">
                    <span className={`inline-block rounded-full px-3 py-1 text-xs font-medium ${topic.color}`}>
                      {topic.difficulty}
                    </span>
                  </div>
                  <CardTitle>{topic.title}</CardTitle>
                  <CardDescription>{topic.description}</CardDescription>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>
      </main>

      {/* Features Section */}
      <section className="mx-auto max-w-7xl px-6 py-16">
        <div className="grid gap-8 md:grid-cols-3">
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary-100">
              <span className="text-2xl">📐</span>
            </div>
            <h3 className="font-semibold text-gray-900">Interactive Math</h3>
            <p className="mt-2 text-sm text-gray-600">
              LaTeX rendering with KaTeX for beautiful equations
            </p>
          </div>
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary-100">
              <span className="text-2xl">📊</span>
            </div>
            <h3 className="font-semibold text-gray-900">Rich Visualizations</h3>
            <p className="mt-2 text-sm text-gray-600">
              Plotly, D3, and Three.js for physics simulations
            </p>
          </div>
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary-100">
              <span className="text-2xl">🎯</span>
            </div>
            <h3 className="font-semibold text-gray-900">Problem Sets</h3>
            <p className="mt-2 text-sm text-gray-600">
              Interactive exercises with instant feedback
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
