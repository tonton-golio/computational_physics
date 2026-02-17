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

interface SimulationProps {
  id: string;
}

export const DYNAMICAL_MODELS_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'hill-function': HillFunction as React.ComponentType<SimulationProps>,
  'gene-expression-noise': GeneExpressionNoise as React.ComponentType<SimulationProps>,
  'binomial-distribution': BinomialDistribution as React.ComponentType<SimulationProps>,
  'poisson-distribution': PoissonDistribution as React.ComponentType<SimulationProps>,
  'michaelis-menten': MichaelisMenten as React.ComponentType<SimulationProps>,
  'steady-state-regulation': SteadyStateRegulation as React.ComponentType<SimulationProps>,
  'lotka-volterra': LotkaVolterraSim as React.ComponentType<SimulationProps>,
  'binomial-poisson-comparison': BinomialPoissonComparison as React.ComponentType<SimulationProps>,
};
