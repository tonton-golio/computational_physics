"use client";

import React from 'react';
import PCAVisualization from './advanced-deep-learning/PCAVisualization';
import ActivationFunctions from './advanced-deep-learning/ActivationFunctions';
import ConvolutionDemo from './advanced-deep-learning/ConvolutionDemo';
import BackpropFlow from './advanced-deep-learning/BackpropFlow';
import LRScheduleComparison from './advanced-deep-learning/LRScheduleComparison';
import RegularizationEffects from './advanced-deep-learning/RegularizationEffects';
import GANTrainingDynamics from './advanced-deep-learning/GANTrainingDynamics';
import UNetArchitecture from './advanced-deep-learning/UNetArchitecture';
import DoubleDescent from './advanced-deep-learning/DoubleDescent';
import LossLandscape from './advanced-deep-learning/LossLandscape';
import {
  AttentionHeatmapDemo,
  LatentInterpolationDemo,
  OptimizerTrajectoryDemo,
} from './advanced-deep-learning/AttentionAndTrainingDemos';
import LRFinder from './advanced-deep-learning/LRFinder';
import ReceptiveFieldGrowth from './advanced-deep-learning/ReceptiveFieldGrowth';
import FilterEvolution from './advanced-deep-learning/FilterEvolution';
import SkipConnectionAblation from './advanced-deep-learning/SkipConnectionAblation';
import AdversarialAttack from './advanced-deep-learning/AdversarialAttack';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const ADVANCED_DEEP_LEARNING_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'adl-pca-demo': PCAVisualization,
  'adl-activation-functions': ActivationFunctions,
  'adl-convolution-demo': ConvolutionDemo,
  'adl-attention-heatmap': AttentionHeatmapDemo,
  'adl-optimizer-trajectories': OptimizerTrajectoryDemo,
  'adl-latent-interpolation': LatentInterpolationDemo,
  'adl-backprop-flow': BackpropFlow,
  'adl-lr-schedule-comparison': LRScheduleComparison,
  'adl-regularization-effects': RegularizationEffects,
  'adl-gan-training-dynamics': GANTrainingDynamics,
  'adl-unet-architecture': UNetArchitecture,
  'adl-double-descent': DoubleDescent,
  'adl-loss-landscape': LossLandscape,
  'adl-lr-finder': LRFinder,
  'adl-receptive-field-growth': ReceptiveFieldGrowth,
  'adl-filter-evolution': FilterEvolution,
  'adl-skip-connection-ablation': SkipConnectionAblation,
  'adl-adversarial-attack': AdversarialAttack,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const ADL_DESCRIPTIONS: Record<string, string> = {
  "adl-pca-demo": "PCA visualization — projecting high-dimensional data onto principal components to reveal dominant variance directions.",
  "adl-activation-functions": "Activation functions — comparing ReLU, sigmoid, tanh, and other nonlinearities used in neural networks.",
  "adl-convolution-demo": "Convolution operation — sliding a learned kernel across an input to produce feature maps in a CNN.",
  "adl-attention-heatmap": "Attention heatmap — visualizing which input tokens a transformer attends to when producing each output.",
  "adl-optimizer-trajectories": "Optimizer trajectories — comparing SGD, Adam, and other optimizers navigating a loss surface.",
  "adl-latent-interpolation": "Latent space interpolation — smoothly morphing between encoded representations in a generative model.",
  "adl-backprop-flow": "Backpropagation flow — tracing gradient propagation layer-by-layer during neural network training.",
  "adl-lr-schedule-comparison": "Learning rate schedules — comparing constant, step decay, cosine annealing, and warmup strategies.",
  "adl-regularization-effects": "Regularization effects — how L1, L2, and dropout penalties affect model weights and generalization.",
  "adl-gan-training-dynamics": "GAN training dynamics — the adversarial interplay between generator and discriminator loss curves.",
  "adl-unet-architecture": "U-Net architecture — the encoder-decoder structure with skip connections used for image segmentation.",
  "adl-double-descent": "Double descent — the phenomenon where test error decreases, rises, then decreases again as model complexity grows.",
  "adl-loss-landscape": "Loss landscape — a topographic view of the optimization surface showing minima, saddle points, and barriers.",
  "adl-lr-finder": "Learning rate finder — sweeping the learning rate from very small to very large to identify the optimal training range.",
  "adl-receptive-field-growth": "Receptive field growth — how stacking convolutional layers expands the input region visible to each neuron.",
  "adl-filter-evolution": "Filter evolution — watching CNN filters transform from random noise into structured edge and texture detectors during training.",
  "adl-skip-connection-ablation": "Skip connection ablation — comparing training dynamics with and without residual connections in deep networks.",
  "adl-adversarial-attack": "Adversarial attack (FGSM) — demonstrating how imperceptibly small input perturbations can fool a neural network classifier.",
};

export const ADL_FIGURES: Record<string, { src: string; caption: string }> = {};
