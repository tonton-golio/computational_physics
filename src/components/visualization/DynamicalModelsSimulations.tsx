"use client";

import React from 'react';
import HillFunction from './dynamical-models/HillFunction';
import GeneExpressionNoise from './dynamical-models/GeneExpressionNoise';
import BinomialDistribution from './dynamical-models/BinomialDistribution';
import PoissonDistribution from './dynamical-models/PoissonDistribution';
import MichaelisMenten from './dynamical-models/MichaelisMenten';
import SteadyStateRegulation from './dynamical-models/SteadyStateRegulation';
import LotkaVolterraSim from './dynamical-models/LotkaVolterraSim';
import BinomialPoissonComparison from './dynamical-models/BinomialPoissonComparison';
import BathtubDynamics from './dynamical-models/BathtubDynamics';
import LuriaDelbruckComparison from './dynamical-models/LuriaDelbruckComparison';
import BifurcationDiagram from './dynamical-models/BifurcationDiagram';
import Repressilator from './dynamical-models/Repressilator';
import ChemotaxisAdaptation from './dynamical-models/ChemotaxisAdaptation';
import ProteomeAllocation from './dynamical-models/ProteomeAllocation';
import PhasePlanePortrait from './dynamical-models/PhasePlanePortrait';
import BacterialLineageTree from './dynamical-models/BacterialLineageTree';
import GillespieTrajectory from './dynamical-models/GillespieTrajectory';
import ProductionDegradationCrossings from './dynamical-models/ProductionDegradationCrossings';
import NotchDeltaCheckerboard from './dynamical-models/NotchDeltaCheckerboard';
import MotifGallery from './dynamical-models/MotifGallery';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const DYNAMICAL_MODELS_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'hill-function': HillFunction,
  'gene-expression-noise': GeneExpressionNoise,
  'binomial-distribution': BinomialDistribution,
  'poisson-distribution': PoissonDistribution,
  'michaelis-menten': MichaelisMenten,
  'steady-state-regulation': SteadyStateRegulation,
  'lotka-volterra': LotkaVolterraSim,
  'binomial-poisson-comparison': BinomialPoissonComparison,
  'bathtub-dynamics': BathtubDynamics,
  'luria-delbruck-comparison': LuriaDelbruckComparison,
  'bifurcation-diagram': BifurcationDiagram,
  'repressilator': Repressilator,
  'chemotaxis-adaptation': ChemotaxisAdaptation,
  'proteome-allocation': ProteomeAllocation,
  'phase-plane-portrait': PhasePlanePortrait,
  'bacterial-lineage-tree': BacterialLineageTree,
  'gillespie-trajectory': GillespieTrajectory,
  'production-degradation-crossings': ProductionDegradationCrossings,
  'notch-delta-checkerboard': NotchDeltaCheckerboard,
  'motif-gallery': MotifGallery,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const DYNAMICAL_DESCRIPTIONS: Record<string, string> = {
  "hill-function": "The Hill function — a sigmoidal response curve modeling cooperative binding in biochemical regulation.",
  "gene-expression-noise": "Gene expression noise — stochastic fluctuations in mRNA and protein levels arising from random biochemical events.",
  "binomial-distribution": "The binomial distribution — probability of k successes in n independent Bernoulli trials with adjustable parameters.",
  "poisson-distribution": "The Poisson distribution — modeling the number of rare events occurring in a fixed interval of time or space.",
  "michaelis-menten": "Michaelis–Menten enzyme kinetics — the saturation curve relating substrate concentration to reaction velocity.",
  "steady-state-regulation": "Steady-state gene regulation — how production and degradation rates determine equilibrium protein concentrations.",
  "lotka-volterra": "Lotka–Volterra predator-prey dynamics — coupled oscillations between predator and prey populations.",
  "binomial-poisson-comparison": "Binomial vs. Poisson comparison — visualizing how the Poisson distribution approximates the binomial for large n and small p.",
  "bathtub-dynamics": "Bathtub dynamics — time-course approach to steady state with adjustable production and degradation rates.",
  "luria-delbruck-comparison": "Luria-Delbrück comparison — side-by-side histograms contrasting directed (Poisson) vs spontaneous mutation models.",
  "bifurcation-diagram": "Bifurcation diagram — interactive hysteresis and saddle-node bifurcations in a positive-feedback gene circuit.",
  "repressilator": "Repressilator — three-gene oscillator showing how odd-numbered repression rings generate sustained oscillations.",
  "chemotaxis-adaptation": "Chemotaxis adaptation — receptor activity and methylation dynamics demonstrating perfect adaptation.",
  "proteome-allocation": "Proteome allocation — interactive pie chart and growth rate curve showing the ribosome-metabolic enzyme tradeoff.",
  "phase-plane-portrait": "Phase plane portrait — interactive 2D phase portrait with nullclines, vector field, and trajectories for a predator-prey system.",
  "bacterial-lineage-tree": "Bacterial lineage tree — visual tree showing stochastic partitioning of molecules across dividing bacterial generations.",
  "gillespie-trajectory": "Gillespie trajectory — stochastic simulation algorithm for a birth-death process, comparing individual trajectories to the ODE mean.",
  "production-degradation-crossings": "Production vs. degradation — graphical construction of steady states by finding where production and degradation curves cross.",
  "notch-delta-checkerboard": "Notch-Delta checkerboard — lateral inhibition simulation showing how a regular checkerboard pattern emerges from random initial conditions.",
  "motif-gallery": "Network motif gallery — interactive gallery of four key network motifs with their step responses and computational personalities.",
};
