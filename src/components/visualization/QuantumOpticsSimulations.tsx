'use client';

import dynamic from 'next/dynamic';
import type { SimulationComponentProps } from '@/shared/types/simulation';


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

const PhaseSpaceExplorer = dynamic(
  () => import('./quantum-optics/PhaseSpaceExplorer'),
  { ssr: false }
);

const JaynesCummingsRevival = dynamic(
  () => import('./quantum-optics/JaynesCummingsRevival'),
  { ssr: false }
);

const VacuumRabiOscillation = dynamic(
  () => import('./quantum-optics/VacuumRabiOscillation'),
  { ssr: false }
);

const TMSVEntanglement = dynamic(
  () => import('./quantum-optics/TMSVEntanglement'),
  { ssr: false }
);

const HomodyneTomography = dynamic(
  () => import('./quantum-optics/HomodyneTomography'),
  { ssr: false }
);

export const QUANTUM_OPTICS_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'wigner-coherent': WignerCoherent,
  'wigner-number-state': WignerNumberState,
  'wigner-squeezed': WignerSqueezed,
  'wigner-cat-state': WignerCatState,
  'phase-space-states': PhaseSpaceStates,
  'phase-space-explorer': PhaseSpaceExplorer,
  'jaynes-cummings-revival': JaynesCummingsRevival,
  'vacuum-rabi-oscillation': VacuumRabiOscillation,
  'tmsv-entanglement': TMSVEntanglement,
  'homodyne-tomography': HomodyneTomography,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const QUANTUM_DESCRIPTIONS: Record<string, string> = {
  "wigner-coherent": "Wigner function of a coherent state — a Gaussian quasi-probability distribution centered at the classical amplitude in phase space.",
  "wigner-number-state": "Wigner function of a Fock (number) state — an oscillating distribution with regions of negativity, a signature of non-classicality.",
  "wigner-squeezed": "Wigner function of a squeezed state — a Gaussian with reduced uncertainty in one quadrature at the cost of increased uncertainty in the other.",
  "wigner-cat-state": "Wigner function of a Schrödinger cat state — a superposition of coherent states showing quantum interference fringes in phase space.",
  "phase-space-states": "Phase space state comparison — side-by-side Wigner function visualizations of different quantum states of light.",
  "phase-space-explorer": "Phase-space explorer — interactive Wigner/Q function heatmap with photon number distribution for vacuum, coherent, squeezed, number, and cat states.",
  "jaynes-cummings-revival": "Jaynes-Cummings collapse & revival — atomic inversion dynamics showing dephasing and rephasing of Rabi oscillations in a coherent cavity field.",
  "vacuum-rabi-oscillation": "Vacuum Rabi oscillation — excited- and ground-state populations for a two-level atom coupled to a single cavity photon, with adjustable coupling and detuning.",
  "tmsv-entanglement": "Two-mode squeezed vacuum entanglement — Duan criterion, thermal photon number, and logarithmic negativity as functions of the squeeze parameter.",
  "homodyne-tomography": "Homodyne tomography — inverse Radon back-projection reconstruction of the Wigner function from simulated quadrature measurements.",
};
