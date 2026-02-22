"use client";

import React from 'react';
import LeastSquaresDemo from './scientific-computing/LeastSquaresDemo';
import ReactionDiffusion from './scientific-computing/ReactionDiffusion';
import LUDecomposition from './scientific-computing/LUDecomposition';
import ConditionNumberDemo from './scientific-computing/ConditionNumberDemo';
import GaussianElimDemo from './scientific-computing/GaussianElimDemo';
import Himmelblau2D from './nonlinear-equations/Himmelblau2D';
import Newton1D from './nonlinear-equations/Newton1D';
import ErrorVsH from './scientific-computing/ErrorVsH';
import GeometricProjection from './scientific-computing/GeometricProjection';
import BasinsOfAttraction from './scientific-computing/BasinsOfAttraction';
import RosenbrockBanana from './scientific-computing/RosenbrockBanana';
import AliasingSlider from './scientific-computing/AliasingSlider';
import StabilityRegions from './scientific-computing/StabilityRegions';
import HeatEquation2D from './scientific-computing/HeatEquation2D';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const SCIENTIFIC_COMPUTING_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'least-squares-demo': LeastSquaresDemo,
  'reaction-diffusion': ReactionDiffusion,
  'lu-decomposition': LUDecomposition,
  'lu-decomp-demo': LUDecomposition,
  'condition-number-demo': ConditionNumberDemo,
  'gaussian-elim-demo': GaussianElimDemo,
  'himmelblau-2d': Himmelblau2D,
  'newton-1d': Newton1D,
  'error-vs-h': ErrorVsH,
  'geometric-projection': GeometricProjection,
  'basins-of-attraction': BasinsOfAttraction,
  'rosenbrock-banana': RosenbrockBanana,
  'aliasing-slider': AliasingSlider,
  'stability-regions': StabilityRegions,
  'heat-equation-2d': HeatEquation2D,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const SCI_DESCRIPTIONS: Record<string, string> = {
  "least-squares-demo": "Least squares fitting — minimizing the sum of squared residuals to find the best-fit curve through data points.",
  "reaction-diffusion": "Reaction-diffusion patterns — Turing-type pattern formation from coupled diffusing and reacting chemicals.",
  "lu-decomposition": "LU decomposition — factoring a matrix into lower and upper triangular components for efficient linear system solving.",
  "lu-decomp-demo": "LU decomposition — step-through factorization of a matrix into lower and upper triangular form.",
  "condition-number-demo": "Condition number — measuring how sensitive a linear system's solution is to perturbations in the input.",
  "gaussian-elim-demo": "Gaussian elimination — row reduction to echelon form for solving systems of linear equations.",
  "himmelblau-2d": "Himmelblau's function — gradient descent and Newton's method on a multimodal 2D optimization landscape.",
  "newton-1d": "Newton's method in 1D — cobweb visualization of root-finding iterations with tangent line steps.",
  "error-vs-h": "Error vs step size — log-log plot showing truncation-rounding tradeoff for forward, centered, and Richardson finite differences.",
  "geometric-projection": "Geometric projection — visualizing least-squares as orthogonal projection of b onto the column space of A.",
  "basins-of-attraction": "Basins of attraction — fractal color map showing which root Newton's method converges to in the complex plane.",
  "rosenbrock-banana": "Rosenbrock banana — contour plot comparing gradient descent and Newton's method on the classic curved-valley test function.",
  "aliasing-slider": "Aliasing demo — interactive sampling of a sine wave showing how under-sampling creates phantom low-frequency waves.",
  "stability-regions": "Stability regions — complex-plane plot of amplification factor boundaries for Forward Euler, Backward Euler, Crank-Nicolson, and RK4.",
  "heat-equation-2d": "2D heat equation — animated FTCS solver showing temperature diffusion from various initial distributions.",
};
