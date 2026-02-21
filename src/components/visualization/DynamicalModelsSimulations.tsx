'use client';

import React from 'react';
import HillFunction from './dynamical-models/HillFunction';
import GeneExpressionNoise from './dynamical-models/GeneExpressionNoise';
import BinomialDistribution from './dynamical-models/BinomialDistribution';
import PoissonDistribution from './dynamical-models/PoissonDistribution';
import MichaelisMenten from './dynamical-models/MichaelisMenten';
import SteadyStateRegulation from './dynamical-models/SteadyStateRegulation';
import { LotkaVolterraSim } from './dynamical-models/LotkaVolterraSim';
import BinomialPoissonComparison from './dynamical-models/BinomialPoissonComparison';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const DYNAMICAL_MODELS_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'hill-function': HillFunction as React.ComponentType<SimulationComponentProps>,
  'gene-expression-noise': GeneExpressionNoise as React.ComponentType<SimulationComponentProps>,
  'binomial-distribution': BinomialDistribution as React.ComponentType<SimulationComponentProps>,
  'poisson-distribution': PoissonDistribution as React.ComponentType<SimulationComponentProps>,
  'michaelis-menten': MichaelisMenten as React.ComponentType<SimulationComponentProps>,
  'steady-state-regulation': SteadyStateRegulation as React.ComponentType<SimulationComponentProps>,
  'lotka-volterra': LotkaVolterraSim as React.ComponentType<SimulationComponentProps>,
  'binomial-poisson-comparison': BinomialPoissonComparison as React.ComponentType<SimulationComponentProps>,
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
};
