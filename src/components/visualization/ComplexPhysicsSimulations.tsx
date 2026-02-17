'use client';

import React from 'react';
import { GameOfLife } from './complex-physics/GameOfLife';
import { LorenzAttractor } from './complex-physics/LorenzAttractor';
import { MandelbrotFractal } from './complex-physics/MandelbrotFractal';
import { IsingModel } from './complex-physics/IsingModel';
import { PercolationSim } from './complex-physics/PercolationSim';
import { BakSneppen } from './complex-physics/BakSneppen';
import { BetHedging } from './complex-physics/BetHedging';
import { HurstExponent } from './complex-physics/HurstExponent';
import { ScaleFreeNetwork } from './complex-physics/ScaleFreeNetwork';
import { PhaseTransitionIsing1D } from './complex-physics/PhaseTransitionIsing1D';
import { BetheLatticePercolation } from './complex-physics/BetheLatticePercolation';
import { FractalDimension } from './complex-physics/FractalDimension';
import { RandomWalkFirstReturn } from './complex-physics/RandomWalkFirstReturn';
import { SandpileModel } from './complex-physics/SandpileModel';
import { StockVariance } from './complex-physics/StockVariance';

interface SimulationProps {
  id?: string;
}

// ============ SIMULATION REGISTRY ============

export const COMPLEX_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'game-of-life': GameOfLife,
  'lorenz-attractor': LorenzAttractor,
  'mandelbrot-fractal': MandelbrotFractal,
  'ising-model': IsingModel,
  'percolation': PercolationSim,
  'bak-sneppen': BakSneppen,
  'bet-hedging': BetHedging,
  'hurst-exponent': HurstExponent,
  'scale-free-network': ScaleFreeNetwork,
  'phase-transition-ising-1d': PhaseTransitionIsing1D,
  'bethe-lattice': BetheLatticePercolation,
  'fractal-dimension': FractalDimension,
  'random-walk-first-return': RandomWalkFirstReturn,
  'sandpile-model': SandpileModel,
  'stock-variance': StockVariance,
};

export function getComplexSimulation(id: string): React.ComponentType<SimulationProps> | null {
  return COMPLEX_SIMULATIONS[id] || null;
}
