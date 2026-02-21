import { listLessonSlugs, readLessonDocument } from "@/infra/content/file-content-repository";

const LANDING_PAGE_PRIORITY = new Map<string, number>([
  ["home", -30],
  ["intro", -20],
  ["introduction", -10],
  ["landingPage", -5],
]);

const TOPIC_LESSON_ORDER: Record<string, string[]> = {
  "complex-physics": [
    "statisticalMechanics",
    "metropolisAlgorithm",
    "phaseTransitions",
    "meanFieldResults",
    "transferMatrix",
    "criticalPhenomena",
    "percolation",
    "fractals",
    "selfOrganizedCriticality",
    "Networks",
    "agentbased",
    "econophysics",
  ],
  "advanced-deep-learning": [
    "ann",
    "cnn",
    "design-and-optimization",
    "vae",
    "gan",
    "unet",
    "transformers",
    "analysis-and-theory",
  ],
  "applied-machine-learning": [
    "loss-optimization",
    "trees-ensembles",
    "neural-networks-fundamentals",
    "dimensionality-reduction",
    "recurrent-neural-networks",
    "graph-neural-networks",
    "generative-adversarial-networks",
  ],
  "applied-statistics": [
    "introduction-concepts",
    "probability-density-functions",
    "error-propagation",
    "simulation-fitting",
    "chi-square-method",
    "hypothesis-testing",
    "anova",
    "design-of-experiments",
    "random-effects",
    "longitudinal-data",
    "bayesian-statistics",
    "advanced-fitting-calibration",
    "machine-learning-data-analysis",
  ],
  "continuum-mechanics": [
    "contapprox",
    "tensorfundamentals",
    "stressandstrain",
    "elasticity",
    "dynamics",
    "fluids",
    "fluidsinmotion",
    "viscousflow",
    "channelsandpipes",
    "gravitywaves",
    "creepingflow",
    "weakstokes",
    "fem",
    "pythonpackages",
    "examdisp",
  ],
  "dynamical-models": [
    "differential-equations",
    "transcription-translation",
    "mutational-analysis",
    "gene-expression-noise",
    "transcriptional-regulation",
    "feedback-loops",
    "signal-transduction",
    "gene-regulatory-networks",
    "bacterial-growth",
  ],
  "inverse-problems": [
    "foundations",
    "regularization",
    "bayesian-inversion",
    "tikhonov",
    "linear-tomography",
    "monte-carlo-methods",
    "geophysical-inversion",
    "information-entropy",
  ],
  "online-reinforcement-learning": [
    "regret",
    "feedback-and-settings",
    "ftl-and-hedge",
    "bandits-ucb-exp3",
    "contextual-bandits-exp4",
    "mdp-and-dp",
    "monte-carlo-rl",
    "td-sarsa-qlearning",
    "deep-rl-dqn",
    "average-reward-online-rl",
  ],
  "quantum-optics": [
    "quantization-single-mode",
    "quantum-fluctuations",
    "multimode-fields",
    "vacuum-fluctuations",
    "coherent-states",
    "coherent-states-field",
    "coherent-states-properties",
    "phase-space-pictures",
    "p-function",
    "wigner-function",
    "coherence-functions",
    "higher-order-coherence",
    "single-photon-experiments",
    "interferometry",
    "squeezed-light",
    "displaced-squeezed-states",
    "atom-field-interaction",
    "jaynes-cummings",
    "cavity-qed",
    "quantum-measurements",
  ],
  "scientific-computing": [
    "bounding-errors",
    "linear-equations",
    "linear-least-squares",
    "nonlinear-equations",
    "nonlinear-systems",
    "optimization",
    "eigen-systems",
    "eigen-algorithms",
    "fft",
    "initial-value-problems",
    "partial-differential-equations",
  ],
};

export const LESSON_SUMMARIES: Record<string, Record<string, string>> = {
  "complex-physics": {
    statisticalMechanics: "Boltzmann distribution, partition functions, and ensemble averages",
    metropolisAlgorithm: "Monte Carlo sampling using the Metropolis-Hastings method",
    phaseTransitions: "Critical behavior and order parameters in phase transitions",
    meanFieldResults: "Mean field approximations for interacting many-body systems",
    transferMatrix: "Exact solutions using transfer matrix techniques",
    criticalPhenomena: "Universality, scaling laws, and renormalization",
    percolation: "Percolation thresholds, critical exponents, and Bethe lattice theory",
    fractals: "Fractal dimension, the Mandelbrot set, and box-counting methods",
    selfOrganizedCriticality: "Sandpile models and power-law distributions",
    Networks: "Graph theory, small-world networks, and scale-free topology",
    agentbased: "Simulating emergent behavior from individual agent rules",
    econophysics: "Statistical physics methods applied to financial markets",
  },
  "advanced-deep-learning": {
    ann: "Feedforward networks, backpropagation, and activation functions",
    cnn: "Convolutional layers, pooling, and image feature extraction",
    "design-and-optimization": "Architecture design choices and hyperparameter tuning",
    vae: "Variational autoencoders for probabilistic latent representations",
    gan: "Generator-discriminator training for realistic data synthesis",
    unet: "Encoder-decoder architecture for image segmentation",
    transformers: "Self-attention mechanisms and sequence modeling",
    "analysis-and-theory": "Theoretical analysis of deep learning convergence and generalization",
  },
  "applied-machine-learning": {
    "loss-optimization": "Gradient descent variants and loss function design",
    "trees-ensembles": "Decision trees, random forests, and boosting methods",
    "dimensionality-reduction": "PCA, t-SNE, and manifold learning techniques",
    "neural-networks-fundamentals": "Basic neural network architectures and training",
    "recurrent-neural-networks": "Sequential data modeling with RNNs and LSTMs",
    "graph-neural-networks": "Message passing and learning on graph-structured data",
    "generative-adversarial-networks": "GAN architectures for generative modeling",
  },
  "applied-statistics": {
    "introduction-concepts": "Core statistical concepts: mean, variance, and distributions",
    "probability-density-functions": "Continuous and discrete probability distributions",
    "error-propagation": "Uncertainty quantification through error propagation rules",
    "simulation-fitting": "Monte Carlo simulation and curve fitting methods",
    "chi-square-method": "Goodness-of-fit testing and chi-square statistics",
    "hypothesis-testing": "Null hypothesis significance testing and p-values",
    anova: "Analysis of variance for comparing group means",
    "design-of-experiments": "Experimental design principles and factorial designs",
    "random-effects": "Mixed models with random and fixed effects",
    "longitudinal-data": "Repeated measures and time-series analysis methods",
    "bayesian-statistics": "Prior distributions, posterior inference, and Bayes' theorem",
    "advanced-fitting-calibration": "Advanced model calibration and nonlinear fitting",
    "machine-learning-data-analysis": "ML techniques for statistical data analysis",
  },
  "continuum-mechanics": {
    contapprox: "The continuum approximation and material descriptions",
    tensorfundamentals: "Tensor algebra, notation, and coordinate transformations",
    stressandstrain: "Stress tensors, strain measures, and constitutive laws",
    elasticity: "Linear elasticity theory and elastic wave propagation",
    dynamics: "Conservation laws and equations of motion for continua",
    fluids: "Fluid statics, pressure, and buoyancy principles",
    fluidsinmotion: "Euler and Navier-Stokes equations for fluid flow",
    viscousflow: "Viscous effects, boundary layers, and Reynolds number",
    channelsandpipes: "Poiseuille flow and flow in channels and pipes",
    gravitywaves: "Surface gravity waves and dispersion relations",
    creepingflow: "Low Reynolds number Stokes flow solutions",
    weakstokes: "Weak formulation of Stokes equations",
    fem: "Finite element method for continuum problems",
    pythonpackages: "Python tools for continuum mechanics simulations",
    examdisp: "Course review — every big idea retold as detective stories",
  },
  "dynamical-models": {
    "differential-equations": "The bathtub equation: how production and degradation set every steady state in biology",
    "transcription-translation": "Coupled ODEs for mRNA and protein — two timescales, one elegant system",
    "mutational-analysis": "DNA proofreading, jackpot cultures, and the statistics of rare events",
    "gene-expression-noise": "Why genetically identical cells behave differently — intrinsic and extrinsic noise",
    "transcriptional-regulation": "The Hill function: how cooperative binding turns a gentle slope into a sharp switch",
    "feedback-loops": "Switches, oscillators, and memory — what feedback loops can do",
    "signal-transduction": "How cells read their mail: chemotaxis, adaptation, and lateral inhibition",
    "gene-regulatory-networks": "The Lego bricks of gene regulation — motifs nature uses over and over",
    "bacterial-growth": "Resource allocation, ribosome regulation, and how a cell turns physics into life",
  },
  "inverse-problems": {
    foundations: "Why inverse problems are hard — ill-posedness, instability, and non-uniqueness",
    regularization: "The L-curve, Tikhonov penalty, and the art of balancing data fit against model simplicity",
    "bayesian-inversion": "Regularization is a prior in disguise — the probabilistic viewpoint that unifies everything",
    tikhonov: "Steepest descent, conjugate gradients, and iterative tricks for million-parameter problems",
    "linear-tomography": "From seismic rays to Earth images — a complete tomographic inversion workflow",
    "monte-carlo-methods": "MCMC sampling — exploring the full posterior when formulas fail",
    "geophysical-inversion": "Fault and glacier case studies where the answer is always a distribution",
    "information-entropy": "Shannon entropy and KL divergence — measuring what the data actually taught you",
  },
  "online-reinforcement-learning": {
    regret: "Regret bounds and online learning performance metrics",
    "feedback-and-settings": "Feedback models: full information vs. bandit",
    "ftl-and-hedge": "Follow the Leader, Hedge, and expert algorithms",
    "bandits-ucb-exp3": "UCB and EXP3 algorithms for bandit problems",
    "contextual-bandits-exp4": "Contextual bandits and the EXP4 algorithm",
    "mdp-and-dp": "Markov decision processes and dynamic programming",
    "monte-carlo-rl": "Monte Carlo methods for policy evaluation",
    "td-sarsa-qlearning": "Temporal difference, SARSA, and Q-learning",
    "deep-rl-dqn": "Deep Q-networks and neural function approximation",
    "average-reward-online-rl": "Average reward criteria for continuing tasks",
  },
  "quantum-optics": {
    "quantization-single-mode": "Canonical quantization of a single electromagnetic mode",
    "quantum-fluctuations": "Vacuum fluctuations and zero-point energy",
    "multimode-fields": "Quantization of multimode electromagnetic fields",
    "vacuum-fluctuations": "Casimir effect and vacuum energy consequences",
    "coherent-states": "Coherent states as quasi-classical quantum states",
    "coherent-states-field": "Coherent state representation of quantum fields",
    "coherent-states-properties": "Displacement operator and Poissonian photon statistics",
    "phase-space-pictures": "Husimi Q-function and phase space representations",
    "p-function": "Glauber-Sudarshan P-representation and its properties",
    "wigner-function": "Wigner quasi-probability distribution and negativity",
    "coherence-functions": "First-order correlation and optical coherence",
    "higher-order-coherence": "Photon bunching, antibunching, and g(2) functions",
    "single-photon-experiments": "Single photon detection and quantum light experiments",
    interferometry: "Mach-Zehnder and quantum-enhanced interferometry",
    "squeezed-light": "Quadrature squeezing and noise reduction below vacuum",
    "displaced-squeezed-states": "Combined displacement and squeezing operations",
    "atom-field-interaction": "Dipole coupling and the rotating wave approximation",
    "jaynes-cummings": "Jaynes-Cummings model and vacuum Rabi oscillations",
    "cavity-qed": "Strong coupling regime in optical cavities",
    "quantum-measurements": "Measurement theory, POVMs, and quantum state tomography",
  },
  "scientific-computing": {
    "bounding-errors": "Floating-point arithmetic and error analysis",
    "linear-equations": "Gaussian elimination and LU factorization",
    "linear-least-squares": "QR factorization and least squares data fitting",
    "nonlinear-equations": "Root finding: bisection, Newton, and secant methods",
    "nonlinear-systems": "Newton's method for systems of nonlinear equations",
    optimization: "BFGS, conjugate gradients, and metaheuristic optimization",
    "eigen-systems": "Power method and Rayleigh quotient iteration",
    "eigen-algorithms": "QR algorithm and eigenvalue decomposition methods",
    fft: "Fast Fourier Transform and spectral analysis",
    "initial-value-problems": "Runge-Kutta and multistep ODE integration methods",
    "partial-differential-equations": "Finite difference and spectral methods for PDEs",
  },
};

export function getLessonSummary(contentId: string, slug: string): string | undefined {
  return LESSON_SUMMARIES[contentId]?.[slug];
}

export function getOrderedLessonSlugs(contentId: string): string[] {
  const slugs = listLessonSlugs(contentId);
  const topicOrder = TOPIC_LESSON_ORDER[contentId];

  if (topicOrder) {
    const orderIndex = new Map(topicOrder.map((slug, i) => [slug, i + 1]));
    return slugs.sort((a, b) => {
      const pa = LANDING_PAGE_PRIORITY.get(a) ?? orderIndex.get(a) ?? 1000;
      const pb = LANDING_PAGE_PRIORITY.get(b) ?? orderIndex.get(b) ?? 1000;
      if (pa !== pb) return pa - pb;
      return a.localeCompare(b);
    });
  }

  return slugs.sort((a, b) => {
    const pa = LANDING_PAGE_PRIORITY.get(a) ?? 0;
    const pb = LANDING_PAGE_PRIORITY.get(b) ?? 0;
    if (pa !== pb) return pa - pb;
    return a.localeCompare(b);
  });
}

export function getLandingPageSlug(contentId: string): string | null {
  const slugs = listLessonSlugs(contentId);
  let best: string | null = null;
  let bestPriority = Infinity;
  for (const slug of slugs) {
    const priority = LANDING_PAGE_PRIORITY.get(slug);
    if (priority !== undefined && priority < bestPriority) {
      best = slug;
      bestPriority = priority;
    }
  }
  return best;
}

export function isLandingPage(slug: string): boolean {
  return LANDING_PAGE_PRIORITY.has(slug);
}

export function getLessonsForTopic(contentId: string): Array<{ slug: string; title: string; summary?: string }> {
  return getOrderedLessonSlugs(contentId)
    .map((slug) => {
      const doc = readLessonDocument(contentId, slug);
      if (!doc) return null;
      const summary = getLessonSummary(contentId, slug);
      return { slug, title: doc.title, ...(summary ? { summary } : {}) };
    })
    .filter((lesson): lesson is { slug: string; title: string; summary?: string } => lesson !== null);
}

export interface SearchableLesson {
  slug: string;
  title: string;
  summary?: string;
  searchableText: string;
}

export function getSearchableLessonsForTopic(contentId: string): SearchableLesson[] {
  return getOrderedLessonSlugs(contentId)
    .map((slug) => {
      const doc = readLessonDocument(contentId, slug);
      if (!doc) return null;

      const summary = getLessonSummary(contentId, slug);
      return {
        slug,
        title: doc.title,
        ...(summary ? { summary } : {}),
        searchableText: `${doc.title}\n${doc.content}`.toLowerCase(),
      };
    })
    .filter((lesson): lesson is SearchableLesson => lesson !== null);
}
