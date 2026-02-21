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
};

export const INVERSE_FIGURES: Record<string, { src: string; caption: string }> = {
  'mri-scan': {
    src: 'https://nci-media.cancer.gov/pdq/media/images/428431-571.jpg',
    caption: 'MRI example from medical imaging.',
  },
  'spect-scan': {
    src: 'https://d2jx2rerrg6sh3.cloudfront.net/image-handler/ts/20170104105121/ri/590/picture/Brain_SPECT_with_Acetazolamide_Slices_thumb.jpg',
    caption: 'SPECT scan example.',
  },
  'seismic-tomography': {
    src: 'http://www.earth.ox.ac.uk/~smachine/cgi/images/welcome_fig_tomo_depth.jpg',
    caption: 'Seismic tomography example.',
  },
  'claude-shannon': {
    src: 'https://d2r55xnwy6nx47.cloudfront.net/uploads/2020/12/Claude-Shannon_2880_Lede.jpg',
    caption: 'Claude Shannon, pioneer of information theory.',
  },
  'climate-grid': {
    src: 'https://caltech-prod.s3.amazonaws.com/main/images/TSchneider-GClimateModel-grid-LES-NEWS-WEB.width-450.jpg',
    caption: 'Model parameterization illustration.',
  },
  'gaussian-process': {
    src: 'https://gowrishankar.info/blog/gaussian-process-and-related-ideas-to-kick-start-bayesian-inference/gp.png',
    caption: 'Gaussian process visualization.',
  },
  'vertical-fault-diagram': {
    src: '/figures/vertical-fault-diagram.svg',
    caption: 'Simplified geometry used in vertical-fault inversion.',
  },
  'glacier-valley-diagram': {
    src: '/figures/glacier-valley-diagram.svg',
    caption: 'Glacier valley cross-section used in thickness inversion.',
  },
};
