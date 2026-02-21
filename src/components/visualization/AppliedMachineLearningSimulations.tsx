'use client';

import React from 'react';
import {
  ExplainedVarianceDemo,
  LossFunctionsDemo,
  PcaCorrelatedDataDemo,
  TreeSplitImpurityDemo,
  ValidationSplitDemo,
} from './applied-machine-learning/AppliedMachineLearningDemos';
import LossLandscapeExplorer from './applied-machine-learning/LossLandscapeExplorer';
import XorEnsembleArena from './applied-machine-learning/XorEnsembleArena';
import SwissRollExplorer from './applied-machine-learning/SwissRollExplorer';
import BackpropBlameGame from './applied-machine-learning/BackpropBlameGame';
import RnnMemoryHighway from './applied-machine-learning/RnnMemoryHighway';
import GnnMessagePassingLive from './applied-machine-learning/GnnMessagePassingLive';
import GanForgerArena from './applied-machine-learning/GanForgerArena';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const APPLIED_MACHINE_LEARNING_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  // Lesson 01 – Loss Functions and Optimization
  'aml-loss-functions': LossFunctionsDemo,
  'aml-loss-landscape': LossLandscapeExplorer,
  'aml-validation-split': ValidationSplitDemo,

  // Lesson 02 – Decision Trees and Ensemble Methods
  'aml-tree-split-impurity': TreeSplitImpurityDemo,
  'aml-tree-ensemble-xor': XorEnsembleArena,

  // Lesson 03 – Dimensionality Reduction
  'aml-pca-correlated-data': PcaCorrelatedDataDemo,
  'aml-explained-variance': ExplainedVarianceDemo,
  'aml-tsne-umap-comparison': SwissRollExplorer,

  // Lesson 04 – Neural Network Fundamentals
  'aml-backprop-blame': BackpropBlameGame,

  // Lesson 05 – Recurrent Neural Networks
  'aml-rnn-memory-highway': RnnMemoryHighway,

  // Lesson 06 – Graph Neural Networks (unified simulation replaces 3 old demos)
  'aml-graph-convolution-intuition': GnnMessagePassingLive,
  'aml-graph-adjacency-demo': GnnMessagePassingLive,
  'aml-graph-message-passing': GnnMessagePassingLive,
  'aml-gnn-message-passing-live': GnnMessagePassingLive,

  // Lesson 07 – Generative Adversarial Networks
  'aml-gan-forger-arena': GanForgerArena,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const AML_DESCRIPTIONS: Record<string, string> = {
  "aml-loss-functions": "Loss functions — comparing MSE, cross-entropy, Huber, and other objective functions for model training.",
  "aml-loss-landscape": "Loss landscape explorer — interactive optimizer on a multi-modal surface with GD, Momentum, RMSProp, and Adam.",
  "aml-validation-split": "Train/validation/test split — 3-way data partitioning with live loss curves showing overfitting dynamics.",
  "aml-tree-split-impurity": "Tree split & impurity — 2D scatter with draggable threshold, real-time Gini impurity and entropy computation.",
  "aml-tree-ensemble-xor": "XOR ensemble arena — single tree vs random forest vs gradient boosting on XOR data with decision boundaries.",
  "aml-pca-correlated-data": "PCA on correlated data — extracting principal components from correlated features to reduce dimensionality.",
  "aml-explained-variance": "Explained variance — the cumulative proportion of total variance captured by successive principal components.",
  "aml-tsne-umap-comparison": "Swiss Roll manifold explorer — PCA vs t-SNE vs UMAP on a 3D Swiss Roll, showing why nonlinear methods win.",
  "aml-backprop-blame": "Backpropagation blame game — animated gradient flow through a 4-layer MLP with vanishing gradient demonstration.",
  "aml-rnn-memory-highway": "RNN vs LSTM memory highway — visualizing information decay in RNNs vs cell-state persistence in LSTMs.",
  "aml-graph-convolution-intuition": "GNN message passing live — interactive graph with step-by-step aggregation and oversmoothing detection.",
  "aml-graph-adjacency-demo": "GNN message passing live — interactive graph with step-by-step aggregation and oversmoothing detection.",
  "aml-graph-message-passing": "GNN message passing live — interactive graph with step-by-step aggregation and oversmoothing detection.",
  "aml-gnn-message-passing-live": "GNN message passing live — interactive graph with step-by-step aggregation and oversmoothing detection.",
  "aml-gan-forger-arena": "GAN forger arena — generator learns a target distribution with live loss curves and mode collapse detection.",
};

export const AML_FIGURES: Record<string, { src: string; caption: string }> = {
  'aml-loss-landscape-3d': {
    src: '/figures/aml-loss-landscape-3d.svg',
    caption: '3D loss landscape with optimizer paths: SGD takes a noisy route, momentum smooths the trajectory, and learning-rate arrows show too-large (bouncing over the valley), too-small (stuck in a shallow pit), and just-right step sizes.',
  },
  'aml-xor-ensemble': {
    src: '/figures/aml-xor-ensemble.svg',
    caption: 'XOR problem solved by an ensemble. Left: a single shallow tree fails with overlapping regions. Right: a random forest produces a clean diagonal decision boundary. Inset: 100 tiny trees voting.',
  },
  'aml-dimred-comparison': {
    src: '/figures/aml-dimred-comparison.svg',
    caption: 'Side-by-side dimensionality reduction: original 3D Swiss-roll dataset, PCA projection (still twisted), and t-SNE/UMAP (beautifully unrolled 2D colors). PCA is linear; t-SNE/UMAP discover the manifold.',
  },
  'aml-backprop-blame': {
    src: '/figures/aml-backprop-blame.svg',
    caption: 'Backpropagation blame-passing: a chain of four neurons (input to output) with red arrows flowing backward labeled "who is to blame?" — fading intensity shows vanishing gradients, while a parallel path with skip connections shows stable flow.',
  },
  'aml-rnn-gradient-flow': {
    src: '/figures/aml-rnn-gradient-flow.svg',
    caption: 'Unrolled RNN diagram from time steps t-3 to t+1. Top: simple RNN with fading gradient arrows. Bottom: LSTM with a bright cell-state highway carrying information across all time steps.',
  },
  'aml-gnn-message-steps': {
    src: '/figures/aml-gnn-message-steps.svg',
    caption: 'Message passing over 3 steps on a 4-node graph (A-B-C-D). Step 0: initial colors. Step 1: 1-hop knowledge. Step 2: 2-hop knowledge. Bonus step 5: all nodes converge to the same dull gray (oversmoothing).',
  },
  'aml-gan-progression': {
    src: '/figures/aml-gan-progression.svg',
    caption: 'GAN training progression. Left: forger starts terrible (noisy blobs). Middle: mode collapse (100 identical smiling faces). Right: healthy GAN (diverse realistic faces). Inset: forger and detective shaking hands at Nash equilibrium.',
  },
};
