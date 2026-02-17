'use client';

import dynamic from 'next/dynamic';

interface SimulationProps {
  id?: string;
}

const WignerCoherent = dynamic(
  () => import('./quantum-optics/WignerCoherent'),
  { ssr: false }
);

const WignerNumberState = dynamic(
  () => import('./quantum-optics/WignerNumberState'),
  { ssr: false }
);

const WignerSqueezed = dynamic(
  () => import('./quantum-optics/WignerSqueezed'),
  { ssr: false }
);

const WignerCatState = dynamic(
  () => import('./quantum-optics/WignerCatState'),
  { ssr: false }
);

const PhaseSpaceStates = dynamic(
  () => import('./quantum-optics/PhaseSpaceStates'),
  { ssr: false }
);

export const QUANTUM_OPTICS_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'wigner-coherent': WignerCoherent,
  'wigner-number-state': WignerNumberState,
  'wigner-squeezed': WignerSqueezed,
  'wigner-cat-state': WignerCatState,
  'phase-space-states': PhaseSpaceStates,
};

export function getQuantumOpticsSimulation(id: string): React.ComponentType<SimulationProps> | null {
  return QUANTUM_OPTICS_SIMULATIONS[id] || null;
}
