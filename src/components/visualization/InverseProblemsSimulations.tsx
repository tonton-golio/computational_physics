'use client';

import React from 'react';
import EntropyDemo from './inverse-problems/EntropyDemo';
import KLDivergence from './inverse-problems/KLDivergence';
import SteepestDescent from './inverse-problems/SteepestDescent';
import LinearTomography from './inverse-problems/LinearTomography';
import MonteCarloIntegration from './inverse-problems/MonteCarloIntegration';
import TikhonovRegularization from './inverse-problems/TikhonovRegularization';
import VerticalFaultMCMC from './inverse-problems/VerticalFaultMCMC';
import GlacierThicknessMCMC from './inverse-problems/GlacierThicknessMCMC';
import SphereInCubeMC from './inverse-problems/SphereInCubeMC';

interface SimulationProps {
  id: string;
}

export const INVERSE_PROBLEMS_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'entropy-demo': EntropyDemo,
  'kl-divergence': KLDivergence,
  'steepest-descent': SteepestDescent,
  'linear-tomography': LinearTomography,
  'monte-carlo-integration': MonteCarloIntegration,
  'tikhonov-regularization': TikhonovRegularization,
  'vertical-fault-mcmc': VerticalFaultMCMC,
  'glacier-thickness-mcmc': GlacierThicknessMCMC,
  'sphere-in-cube-mc': SphereInCubeMC,
};
