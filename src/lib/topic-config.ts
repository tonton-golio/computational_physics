interface TopicConfig {
  title: string;
  description: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced" | "Expert";
  color: string;
  relatedTopics?: string[];
}

export const TOPICS = {
  "quantum-optics": {
    title: "Quantum Optics",
    description: "Wigner functions, phase space dynamics, and quantum state visualization",
    difficulty: "Expert",
    color: "bg-purple-100 text-purple-800",
    relatedTopics: ["complex-physics", "scientific-computing", "advanced-deep-learning"],
  },
  "continuum-mechanics": {
    title: "Continuum Mechanics",
    description: "Rod dynamics, iceberg simulations, and gravity waves",
    difficulty: "Expert",
    color: "bg-blue-100 text-blue-800",
    relatedTopics: ["dynamical-models", "scientific-computing", "inverse-problems", "complex-physics"],
  },
  "inverse-problems": {
    title: "Inverse Problems",
    description: "Tikhonov regularization and practical inversion techniques",
    difficulty: "Advanced",
    color: "bg-green-100 text-green-800",
    relatedTopics: ["applied-statistics", "scientific-computing", "continuum-mechanics", "dynamical-models"],
  },
  "complex-physics": {
    title: "Complex Physics",
    description: "Statistical mechanics, percolation, and fractals",
    difficulty: "Intermediate",
    color: "bg-orange-100 text-orange-800",
    relatedTopics: ["dynamical-models", "quantum-optics", "continuum-mechanics", "applied-statistics"],
  },
  "scientific-computing": {
    title: "Scientific Computing",
    description: "Linear equations, optimization, and numerical methods",
    difficulty: "Intermediate",
    color: "bg-cyan-100 text-cyan-800",
    relatedTopics: ["applied-statistics", "inverse-problems", "continuum-mechanics", "advanced-deep-learning", "online-reinforcement-learning"],
  },
  "online-reinforcement-learning": {
    title: "Online Reinforcement Learning",
    description: "Multi-armed bandits and regret analysis",
    difficulty: "Intermediate",
    color: "bg-pink-100 text-pink-800",
    relatedTopics: ["advanced-deep-learning", "applied-statistics", "scientific-computing", "dynamical-models"],
  },
  "advanced-deep-learning": {
    title: "Advanced Deep Learning",
    description: "GANs, VAEs, CNNs, and U-Net architectures",
    difficulty: "Advanced",
    color: "bg-red-100 text-red-800",
    relatedTopics: ["online-reinforcement-learning", "scientific-computing", "applied-statistics", "quantum-optics"],
  },
  "applied-statistics": {
    title: "Applied Statistics",
    description: "Statistical methods and data analysis techniques",
    difficulty: "Beginner",
    color: "bg-yellow-100 text-yellow-800",
    relatedTopics: ["scientific-computing", "inverse-problems", "online-reinforcement-learning", "advanced-deep-learning", "complex-physics"],
  },
  "applied-machine-learning": {
    title: "Applied Machine Learning",
    description: "Loss optimization, ensemble methods, and representation learning",
    difficulty: "Intermediate",
    color: "bg-indigo-100 text-indigo-800",
    relatedTopics: ["advanced-deep-learning", "applied-statistics", "scientific-computing", "online-reinforcement-learning", "complex-physics"],
  },
  "dynamical-models": {
    title: "Dynamical Models",
    description: "Mathematical modeling of dynamic systems",
    difficulty: "Beginner",
    color: "bg-teal-100 text-teal-800",
    relatedTopics: ["continuum-mechanics", "complex-physics", "inverse-problems", "online-reinforcement-learning"],
  },
} as const satisfies Record<string, TopicConfig>;

export type TopicSlug = keyof typeof TOPICS;
