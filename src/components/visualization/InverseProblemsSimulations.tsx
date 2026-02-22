"use client";

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
import PriorLikelihoodPosterior from './inverse-problems/PriorLikelihoodPosterior';
import ConjugateGradientRace from './inverse-problems/ConjugateGradientRace';
import LCurveConstruction from './inverse-problems/LCurveConstruction';
import RaySmearingExplorer from './inverse-problems/RaySmearingExplorer';
import PosteriorWalkerArena from './inverse-problems/PosteriorWalkerArena';
import TradeoffCloud from './inverse-problems/TradeoffCloud';
import EntropyKLCalculator from './inverse-problems/EntropyKLCalculator';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const INVERSE_PROBLEMS_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'entropy-demo': EntropyDemo,
  'kl-divergence': KLDivergence,
  'steepest-descent': SteepestDescent,
  'linear-tomography': LinearTomography,
  'monte-carlo-integration': MonteCarloIntegration,
  'tikhonov-regularization': TikhonovRegularization,
  'vertical-fault-mcmc': VerticalFaultMCMC,
  'glacier-thickness-mcmc': GlacierThicknessMCMC,
  'sphere-in-cube-mc': SphereInCubeMC,
  'prior-likelihood-posterior': PriorLikelihoodPosterior,
  'conjugate-gradient-race': ConjugateGradientRace,
  'l-curve-construction': LCurveConstruction,
  'ray-smearing-explorer': RaySmearingExplorer,
  'posterior-walker-arena': PosteriorWalkerArena,
  'tradeoff-cloud': TradeoffCloud,
  'entropy-kl-calculator': EntropyKLCalculator,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const INVERSE_DESCRIPTIONS: Record<string, string> = {
  "entropy-demo": "Shannon entropy — measuring the information content of a probability distribution and how it changes with uniformity.",
  "kl-divergence": "Kullback–Leibler divergence — quantifying how one probability distribution differs from a reference distribution.",
  "steepest-descent": "Steepest descent optimization — iteratively minimizing a function by following the negative gradient direction.",
  "linear-tomography": "Linear tomography — reconstructing an internal structure from projection data using Tikhonov regularization.",
  "monte-carlo-integration": "Monte Carlo integration — estimating definite integrals by random sampling and averaging function evaluations.",
  "tikhonov-regularization": "Tikhonov regularization — stabilizing ill-posed inverse problems by adding a penalty term to the objective function.",
  "vertical-fault-mcmc": "Vertical fault inversion with MCMC — sampling the posterior distribution of fault parameters using Markov Chain Monte Carlo.",
  "glacier-thickness-mcmc": "Glacier thickness estimation with MCMC — inferring ice depth from surface observations using Bayesian inversion.",
  "sphere-in-cube-mc": "Sphere-in-cube Monte Carlo — estimating the volume ratio by randomly sampling points inside a cube.",
  "prior-likelihood-posterior": "Bayesian update — visualizing how a Gaussian prior and likelihood combine into a posterior distribution.",
  "conjugate-gradient-race": "Conjugate gradients vs steepest descent — comparing convergence on an elliptical quadratic objective.",
  "l-curve-construction": "L-curve method — sweeping regularization strength and finding the optimal corner that balances data fit against model complexity.",
  "ray-smearing-explorer": "Ray smearing explorer — visualizing how seismic ray geometries sample a tomographic model and where coverage gaps create blind spots.",
  "posterior-walker-arena": "Posterior walker arena — MCMC walkers exploring a bimodal 2D posterior, showing burn-in, convergence, and mode-switching behavior.",
  "tradeoff-cloud": "Trade-off cloud — depth-slip anti-correlation in a 2-parameter inverse problem, showing how noise stretches the posterior along trade-off directions.",
  "entropy-kl-calculator": "Entropy and divergence calculator — interactive computation of Shannon entropy, KL divergence (both directions), cross-entropy, and Jensen-Shannon divergence for two discrete distributions.",
};

export const INVERSE_FIGURES: Record<string, { src: string; caption: string }> = {
  'vertical-fault-diagram': {
    src: '/figures/vertical-fault-diagram.svg',
    caption: 'Simplified geometry used in vertical-fault inversion.',
  },
  'glacier-valley-diagram': {
    src: '/figures/glacier-valley-diagram.svg',
    caption: 'Glacier valley cross-section used in thickness inversion.',
  },
};
