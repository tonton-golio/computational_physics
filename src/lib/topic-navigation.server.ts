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
    "percolation_and_fractals",
    "selfOrganizedCriticality",
    "Networks",
    "agentbased",
    "econophysics",
  ],
  "advanced-deep-learning": [
    "ann",
    "cnn",
    "unet",
    "vae",
    "gan",
    "transformers",
    "design-and-optimization",
    "analysis-and-theory",
  ],
  "applied-machine-learning": [
    "loss-optimization",
    "trees-ensembles",
    "dimensionality-reduction",
    "graph-neural-networks",
    "recurrent-neural-networks",
    "generative-adversarial-networks",
  ],
  "applied-statistics": [
    "introduction-concepts",
    "probability-density-functions",
    "error-propagation",
    "chi-square-method",
    "hypothesis-testing",
    "simulation-fitting",
    "anova",
    "design-of-experiments",
    "random-effects",
    "longitudinal-data",
    "bayesian-statistics",
    "advanced-fitting-calibration",
    "machine-learning-data-analysis",
  ],
  "continuum-mechanics": [
    "frontpagetext",
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
    "coursequote",
    "notebook",
  ],
  "dynamical-models": [
    "gene-expression-noise",
    "differential-equations",
    "transcription-translation",
    "transcriptional-regulation",
    "feedback-loops",
    "gene-regulatory-networks",
    "signal-transduction",
    "mutational-analysis",
    "bacterial-growth",
  ],
  "inverse-problems": [
    "intro",
    "week1",
    "week2",
    "week3",
    "week4",
    "Tikonov",
    "linear_tomography",
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
    "assignments-projects",
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
    "papers",
  ],
  "scientific-computing": [
    "bounding_errors",
    "linear_equations",
    "linearLeastSquares",
    "nonlinearEquationsOptimization",
    "nonlinearSystems",
    "optimization",
    "eigenSystems",
    "eigenAlgorithms",
    "initialValueProblems",
    "partialDifferentialEquations",
    "fft",
  ],
};

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

export function getLessonsForTopic(contentId: string): Array<{ slug: string; title: string }> {
  return getOrderedLessonSlugs(contentId)
    .map((slug) => {
      const doc = readLessonDocument(contentId, slug);
      return doc ? { slug, title: doc.title } : null;
    })
    .filter((lesson): lesson is { slug: string; title: string } => lesson !== null);
}

export interface SearchableLesson {
  slug: string;
  title: string;
  searchableText: string;
}

export function getSearchableLessonsForTopic(contentId: string): SearchableLesson[] {
  return getOrderedLessonSlugs(contentId)
    .map((slug) => {
      const doc = readLessonDocument(contentId, slug);
      if (!doc) return null;

      return {
        slug,
        title: doc.title,
        searchableText: `${doc.title}\n${doc.content}`.toLowerCase(),
      };
    })
    .filter((lesson): lesson is SearchableLesson => lesson !== null);
}
