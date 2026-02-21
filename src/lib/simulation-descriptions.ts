/**
 * Aggregator — merges co-located description exports from each topic registry.
 * Individual descriptions live alongside their simulation registries in
 * src/components/visualization/*Simulations.tsx files.
 */
import { COMPLEX_DESCRIPTIONS } from '@/components/visualization/ComplexPhysicsSimulations';
import { INVERSE_DESCRIPTIONS } from '@/components/visualization/InverseProblemsSimulations';
import { QUANTUM_DESCRIPTIONS } from '@/components/visualization/QuantumOpticsSimulations';
import { DYNAMICAL_DESCRIPTIONS } from '@/components/visualization/DynamicalModelsSimulations';
import { CONTINUUM_DESCRIPTIONS } from '@/components/visualization/ContinuumMechanicsSimulations';
import { ADL_DESCRIPTIONS } from '@/components/visualization/AdvancedDeepLearningSimulations';
import { AML_DESCRIPTIONS } from '@/components/visualization/AppliedMachineLearningSimulations';
import { SCI_DESCRIPTIONS } from '@/components/visualization/ScientificComputingSimulations';
import { ORL_DESCRIPTIONS } from '@/components/visualization/OnlineReinforcementSimulations';
import { STATS_DESCRIPTIONS } from '@/components/visualization/AppliedStatsSimulations';
import { EIGEN_DESCRIPTIONS } from '@/components/visualization/EigenvalueSimulations';

export const SIMULATION_DESCRIPTIONS: Record<string, string> = {
  ...COMPLEX_DESCRIPTIONS,
  ...INVERSE_DESCRIPTIONS,
  ...QUANTUM_DESCRIPTIONS,
  ...DYNAMICAL_DESCRIPTIONS,
  ...CONTINUUM_DESCRIPTIONS,
  ...ADL_DESCRIPTIONS,
  ...AML_DESCRIPTIONS,
  ...SCI_DESCRIPTIONS,
  ...ORL_DESCRIPTIONS,
  ...STATS_DESCRIPTIONS,
  ...EIGEN_DESCRIPTIONS,
};

/**
 * Returns a human-readable title derived from a simulation ID.
 * Converts kebab-case to Title Case (e.g., "steepest-descent" → "Steepest Descent").
 */
export function simulationTitle(id: string): string {
  return id
    .replace(/^(adl|aml)-/, "") // strip topic prefixes
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
