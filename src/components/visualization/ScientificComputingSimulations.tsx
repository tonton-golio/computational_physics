'use client';

import React from 'react';
import { LeastSquaresDemo } from './scientific-computing/LeastSquaresDemo';
import { ReactionDiffusion } from './scientific-computing/ReactionDiffusion';
import { LUDecomposition } from './scientific-computing/LUDecomposition';
import Himmelblau2D from './nonlinear-equations/Himmelblau2D';
import Newton1D from './nonlinear-equations/Newton1D';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const SCIENTIFIC_COMPUTING_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'least-squares-demo': LeastSquaresDemo,
  'reaction-diffusion': ReactionDiffusion,
  'lu-decomposition': LUDecomposition,
  'himmelblau-2d': Himmelblau2D,
  'newton-1d': Newton1D,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const SCI_DESCRIPTIONS: Record<string, string> = {
  "least-squares-demo": "Least squares fitting — minimizing the sum of squared residuals to find the best-fit curve through data points.",
  "reaction-diffusion": "Reaction-diffusion patterns — Turing-type pattern formation from coupled diffusing and reacting chemicals.",
  "lu-decomposition": "LU decomposition — factoring a matrix into lower and upper triangular components for efficient linear system solving.",
  "himmelblau-2d": "Himmelblau's function — gradient descent and Newton's method on a multimodal 2D optimization landscape.",
  "newton-1d": "Newton's method in 1D — cobweb visualization of root-finding iterations with tangent line steps.",
};
