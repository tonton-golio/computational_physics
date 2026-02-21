/**
 * Aggregator — merges co-located figure exports from each topic registry.
 * Individual figure definitions live alongside their simulation registries in
 * src/components/visualization/*Simulations.tsx files.
 */
import { COMPLEX_FIGURES } from '@/components/visualization/ComplexPhysicsSimulations';
import { INVERSE_FIGURES } from '@/components/visualization/InverseProblemsSimulations';
import { ADL_FIGURES } from '@/components/visualization/AdvancedDeepLearningSimulations';
import { AML_FIGURES } from '@/components/visualization/AppliedMachineLearningSimulations';
import { ORL_FIGURES } from '@/components/visualization/OnlineReinforcementSimulations';
import { EIGEN_FIGURES } from '@/components/visualization/EigenvalueSimulations';

export const FIGURE_DEFS: Record<string, { src: string; caption: string }> = {
  ...COMPLEX_FIGURES,
  ...INVERSE_FIGURES,
  ...ADL_FIGURES,
  ...AML_FIGURES,
  ...ORL_FIGURES,
  ...EIGEN_FIGURES,
};
