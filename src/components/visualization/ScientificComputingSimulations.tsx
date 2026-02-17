'use client';

import React from 'react';
import { LeastSquaresDemo } from './scientific-computing/LeastSquaresDemo';
import { ReactionDiffusion } from './scientific-computing/ReactionDiffusion';
import { LUDecomposition } from './scientific-computing/LUDecomposition';

interface SimulationProps {
  id?: string;
}

export const SCIENTIFIC_COMPUTING_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'least-squares-demo': LeastSquaresDemo,
  'reaction-diffusion': ReactionDiffusion,
  'lu-decomposition': LUDecomposition,
};

export function getScientificComputingSimulation(id: string): React.ComponentType<SimulationProps> | null {
  return SCIENTIFIC_COMPUTING_SIMULATIONS[id] || null;
}
