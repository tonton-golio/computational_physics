import type { SimulationDefinition, SimulationComponent } from "@/shared/types/simulation";
import type { TopicId } from "@/lib/topic-config";

type SimulationRegistry = Record<string, SimulationComponent>;

interface RegistryGroup {
  topic: TopicId;
  title: string;
  load: () => Promise<SimulationRegistry>;
}

const registryCache = new Map<number, Promise<SimulationRegistry>>();
const resolvedDefinitionCache = new Map<string, SimulationDefinition | null>();
const inFlightDefinitionCache = new Map<string, Promise<SimulationDefinition | null>>();
// Fast-path index: simulation id → registryGroups index. Populated only from
// registries that have actually been loaded, so it never contains guessed ids.
// A miss simply falls back to the full scan below, preserving correctness.
const idToGroupIndex = new Map<string, number>();

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
    topic: "scientific-computing",
    title: "Eigenvalue Visualizations",
    load: () => import("@/components/visualization/EigenvalueSimulations").then((m) => m.EIGENVALUE_SIMULATIONS as SimulationRegistry),
  },
];

function loadRegistry(groupIndex: number, group: RegistryGroup): Promise<SimulationRegistry> {
  const cached = registryCache.get(groupIndex);
  if (cached) return cached;
  const promise = group.load().then((registry) => {
    // Record every id this registry exposes so future lookups can take the
    // fast path (load only this one chunk) instead of scanning all groups.
    for (const registryId of Object.keys(registry)) {
      if (!idToGroupIndex.has(registryId)) idToGroupIndex.set(registryId, groupIndex);
    }
    return registry;
  });
  promise.catch(() => registryCache.delete(groupIndex));
  registryCache.set(groupIndex, promise);
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
  // Fast path: if we already know which registry group owns this id, load only
  // that one chunk instead of scanning (and importing) every topic registry.
  const knownIndex = idToGroupIndex.get(id);
  if (knownIndex !== undefined) {
    const group = registryGroups[knownIndex];
    try {
      const registry = await loadRegistry(knownIndex, group);
      const component = registry[id];
      if (component) return makeDefinition(group, id, component);
    } catch {
      // Fall through to the full scan below if the targeted chunk fails.
    }
  }

  // Fallback full scan — also the only path the first time an id is seen.
  for (let i = 0; i < registryGroups.length; i++) {
    const group = registryGroups[i];
    try {
      const registry = await loadRegistry(i, group);
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
