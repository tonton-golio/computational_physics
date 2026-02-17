'use client';

import React from 'react';
import ConcentrationBounds from './online-reinforcement/ConcentrationBounds';
import BernoulliTrials from './online-reinforcement/BernoulliTrials';
import MultiArmedBandit from './online-reinforcement/MultiArmedBandit';
import MDPSimulation from './online-reinforcement/MDPSimulation';
import ActivationFunctions from './online-reinforcement/ActivationFunctions';
import StochasticApproximation from './online-reinforcement/StochasticApproximation';
import CartPoleLearningCurves from './online-reinforcement/CartPoleLearningCurves';
import RegretGrowthComparison from './online-reinforcement/RegretGrowthComparison';
import FTLInstability from './online-reinforcement/FTLInstability';
import HedgeWeightsRegret from './online-reinforcement/HedgeWeightsRegret';
import BanditRegretComparison from './online-reinforcement/BanditRegretComparison';
import ContextualBanditExp4 from './online-reinforcement/ContextualBanditExp4';
import GridworldMDP from './online-reinforcement/GridworldMDP';
import DPConvergence from './online-reinforcement/DPConvergence';
import MonteCarloConvergence from './online-reinforcement/MonteCarloConvergence';
import SarsaVsQLearning from './online-reinforcement/SarsaVsQLearning';
import DQNStability from './online-reinforcement/DQNStability';
import AverageRewardVsDiscounted from './online-reinforcement/AverageRewardVsDiscounted';

interface SimulationProps {
  id: string;
}

export const ONLINE_REINFORCEMENT_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'concentration-bounds': ConcentrationBounds,
  'bernoulli-trials': BernoulliTrials,
  'multi-armed-bandit': MultiArmedBandit,
  'mdp-simulation': MDPSimulation,
  'activation-functions': ActivationFunctions,
  'stochastic-approximation': StochasticApproximation,
  'cartpole-learning-curves': CartPoleLearningCurves,
  'regret-growth-comparison': RegretGrowthComparison,
  'ftl-instability': FTLInstability,
  'hedge-weights-regret': HedgeWeightsRegret,
  'bandit-regret-comparison': BanditRegretComparison,
  'contextual-bandit-exp4': ContextualBanditExp4,
  'gridworld-mdp': GridworldMDP,
  'dp-convergence': DPConvergence,
  'monte-carlo-convergence': MonteCarloConvergence,
  'sarsa-vs-qlearning': SarsaVsQLearning,
  'dqn-stability': DQNStability,
  'average-reward-vs-discounted': AverageRewardVsDiscounted,
};
