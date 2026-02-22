"use client";

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
import BlackjackTrajectory from './online-reinforcement/BlackjackTrajectory';
import CliffWalking from './online-reinforcement/CliffWalking';
import ReplayBufferExplorer from './online-reinforcement/ReplayBufferExplorer';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const ONLINE_REINFORCEMENT_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
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
  'blackjack-trajectory': BlackjackTrajectory,
  'cliff-walking': CliffWalking,
  'replay-buffer-explorer': ReplayBufferExplorer,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const ORL_DESCRIPTIONS: Record<string, string> = {
  "concentration-bounds": "Concentration inequalities — Hoeffding, Bernstein, and other bounds on how random variables deviate from their mean.",
  "bernoulli-trials": "Bernoulli trials — repeated independent experiments with two outcomes, illustrating the law of large numbers.",
  "multi-armed-bandit": "Multi-armed bandit — balancing exploration and exploitation across slot machines with unknown reward distributions.",
  "mdp-simulation": "Markov Decision Process — an agent navigating states with probabilistic transitions and rewards.",
  "activation-functions": "Activation functions — visualizing nonlinear transformations used in neural network layers.",
  "stochastic-approximation": "Stochastic approximation — iterative algorithms converging to a fixed point using noisy gradient estimates.",
  "cartpole-learning-curves": "CartPole learning curves — tracking reward improvement as an RL agent learns to balance a pole.",
  "regret-growth-comparison": "Regret growth comparison — how cumulative regret scales over time for different bandit algorithms.",
  "ftl-instability": "Follow-the-Leader instability — demonstrating why greedy strategies fail in adversarial online settings.",
  "hedge-weights-regret": "Hedge algorithm — multiplicative weight updates achieving logarithmic regret in adversarial prediction.",
  "bandit-regret-comparison": "Bandit regret comparison — UCB, Thompson Sampling, and epsilon-greedy performance side-by-side.",
  "contextual-bandit-exp4": "Contextual bandit (EXP4) — extending multi-armed bandits with side information for personalized decisions.",
  "gridworld-mdp": "Gridworld MDP — a grid environment where an agent learns optimal navigation through value iteration or policy iteration.",
  "dp-convergence": "Dynamic programming convergence — watching value iteration converge to the optimal value function.",
  "monte-carlo-convergence": "Monte Carlo convergence — estimating value functions from sampled episode returns.",
  "sarsa-vs-qlearning": "SARSA vs. Q-Learning — comparing on-policy and off-policy temporal-difference methods.",
  "dqn-stability": "Deep Q-Network stability — how experience replay and target networks stabilize deep RL training.",
  "average-reward-vs-discounted": "Average reward vs. discounted — comparing two objective formulations for continuing RL tasks.",
  "blackjack-trajectory": "Blackjack Monte Carlo — first-visit MC value estimation on simplified Blackjack, showing how state values converge from complete episodes.",
  "cliff-walking": "Cliff Walking — SARSA takes the safe path while Q-learning finds the optimal cliff-edge route, illustrating on-policy vs off-policy TD control.",
  "replay-buffer-explorer": "Experience Replay Buffer — visualizing how random sampling from a circular buffer breaks temporal correlation and stabilizes DQN training.",
};

export const ORL_FIGURES: Record<string, { src: string; caption: string }> = {};
