import type { SimulationDefinition, SimulationComponent } from "@/shared/types/simulation";

type SimulationRegistry = Record<string, SimulationComponent>;

interface RegistryGroup {
  topic: string;
  title: string;
  load: () => Promise<SimulationRegistry>;
}

const registryCache = new Map<string, Promise<SimulationRegistry>>();
const resolvedDefinitionCache = new Map<string, SimulationDefinition | null>();
const inFlightDefinitionCache = new Map<string, Promise<SimulationDefinition | null>>();

const registryGroups: RegistryGroup[] = [
  {
    topic: "applied-statistics",
    title: "Applied Statistics",
    load: () =>
      import("@/components/visualization/AppliedStatsSimulations").then((m) => m.APPLIED_STATS_SIMULATIONS as SimulationRegistry),
  },
  {
    topic: "complex-physics",
    title: "Complex Physics",
    load: () => import("@/components/visualization/ComplexPhysicsSimulations").then((m) => m.COMPLEX_SIMULATIONS as SimulationRegistry),
  },
  {
    topic: "continuum-mechanics",
    title: "Continuum Mechanics",
    load: () =>
      import("@/components/visualization/ContinuumMechanicsSimulations").then(
        (m) => m.CONTINUUM_MECHANICS_SIMULATIONS as SimulationRegistry
      ),
  },
  {
    topic: "inverse-problems",
    title: "Inverse Problems",
    load: () =>
      import("@/components/visualization/InverseProblemsSimulations").then((m) => m.INVERSE_PROBLEMS_SIMULATIONS as SimulationRegistry),
  },
  {
    topic: "quantum-optics",
    title: "Quantum Optics",
    load: () =>
      import("@/components/visualization/QuantumOpticsSimulations").then((m) => m.QUANTUM_OPTICS_SIMULATIONS as SimulationRegistry),
  },
  {
    topic: "dynamical-models",
    title: "Dynamical Models",
    load: () =>
      import("@/components/visualization/DynamicalModelsSimulations").then((m) => m.DYNAMICAL_MODELS_SIMULATIONS as SimulationRegistry),
  },
  {
    topic: "advanced-deep-learning",
    title: "Advanced Deep Learning",
    load: () =>
      import("@/components/visualization/AdvancedDeepLearningSimulations").then(
        (m) => m.ADVANCED_DEEP_LEARNING_SIMULATIONS as SimulationRegistry
      ),
  },
  {
    topic: "applied-machine-learning",
    title: "Applied Machine Learning",
    load: () =>
      import("@/components/visualization/AppliedMachineLearningSimulations").then(
        (m) => m.APPLIED_MACHINE_LEARNING_SIMULATIONS as SimulationRegistry
      ),
  },
  {
    topic: "scientific-computing",
    title: "Scientific Computing",
    load: () =>
      import("@/components/visualization/ScientificComputingSimulations").then(
        (m) => m.SCIENTIFIC_COMPUTING_SIMULATIONS as SimulationRegistry
      ),
  },
  {
    topic: "online-reinforcement-learning",
    title: "Online Reinforcement Learning",
    load: () =>
      import("@/components/visualization/OnlineReinforcementSimulations").then(
        (m) => m.ONLINE_REINFORCEMENT_SIMULATIONS as SimulationRegistry
      ),
  },
  {
    topic: "linear-algebra",
    title: "Eigenvalue Visualizations",
    load: () => import("@/components/visualization/EigenvalueSimulations").then((m) => m.EIGENVALUE_SIMULATIONS as SimulationRegistry),
  },
];

const ONLINE_REINFORCEMENT_IDS = new Set<string>([
  "concentration-bounds",
  "bernoulli-trials",
  "multi-armed-bandit",
  "mdp-simulation",
  "activation-functions",
  "stochastic-approximation",
  "cartpole-learning-curves",
  "regret-growth-comparison",
  "ftl-instability",
  "hedge-weights-regret",
  "bandit-regret-comparison",
  "contextual-bandit-exp4",
  "gridworld-mdp",
  "dp-convergence",
  "monte-carlo-convergence",
  "sarsa-vs-qlearning",
  "dqn-stability",
  "average-reward-vs-discounted",
]);

function loadRegistry(group: RegistryGroup): Promise<SimulationRegistry> {
  const cached = registryCache.get(group.topic);
  if (cached) return cached;
  const promise = group.load();
  registryCache.set(group.topic, promise);
  return promise;
}

function makeDefinition(group: RegistryGroup, id: string, component: SimulationComponent): SimulationDefinition {
  return {
    id,
    topic: group.topic,
    title: `${group.title}: ${id}`,
    load: async () => component,
    fallbackToGraph: true,
  };
}

export async function resolveSimulationDefinition(id: string): Promise<SimulationDefinition | null> {
  const cachedDefinition = resolvedDefinitionCache.get(id);
  if (cachedDefinition !== undefined) return cachedDefinition;

  const inFlightDefinition = inFlightDefinitionCache.get(id);
  if (inFlightDefinition) return inFlightDefinition;

  const lookupPromise = (async (): Promise<SimulationDefinition | null> => {
  // Fast path: online-reinforcement pages can render many placeholders at once.
  // Avoid scanning/importing all registry groups before reaching this topic.
  if (ONLINE_REINFORCEMENT_IDS.has(id)) {
    const onlineGroup = registryGroups.find((group) => group.topic === "online-reinforcement-learning");
    if (onlineGroup) {
      try {
        const registry = await loadRegistry(onlineGroup);
        const component = registry[id];
        if (component) return makeDefinition(onlineGroup, id, component);
      } catch {
        // Fall through to general lookup.
      }
    }
  }

  for (const group of registryGroups) {
    try {
      const registry = await loadRegistry(group);
      const component = registry[id];
      if (!component) continue;
      return makeDefinition(group, id, component);
    } catch {
      // Isolate faulty registry imports so one broken module does not break all simulations.
      continue;
    }
  }
  return null;
  })();

  inFlightDefinitionCache.set(id, lookupPromise);
  try {
    const definition = await lookupPromise;
    resolvedDefinitionCache.set(id, definition);
    return definition;
  } finally {
    inFlightDefinitionCache.delete(id);
  }
}

export async function prefetchSimulationDefinition(id: string): Promise<void> {
  try {
    await resolveSimulationDefinition(id);
  } catch {
    // Best-effort prefetch only; loading path handles errors.
  }
}
