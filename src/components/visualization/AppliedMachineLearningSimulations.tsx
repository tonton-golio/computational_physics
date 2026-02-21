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
import LossCurvesOutliers from './applied-machine-learning/LossCurvesOutliers';
import TreeGrowthSteps from './applied-machine-learning/TreeGrowthSteps';
import ForestVsSingleTree from './applied-machine-learning/ForestVsSingleTree';
import ForwardBackwardPass from './applied-machine-learning/ForwardBackwardPass';
import ActivationGallery from './applied-machine-learning/ActivationGallery';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export const APPLIED_MACHINE_LEARNING_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  // Lesson 01 – Loss Functions and Optimization
  'aml-loss-functions': LossFunctionsDemo,
  'aml-loss-landscape': LossLandscapeExplorer,
  'aml-validation-split': ValidationSplitDemo,
  'aml-loss-curves-outliers': LossCurvesOutliers,

  // Lesson 02 – Decision Trees and Ensemble Methods
  'aml-tree-split-impurity': TreeSplitImpurityDemo,
  'aml-tree-ensemble-xor': XorEnsembleArena,
  'aml-tree-growth-steps': TreeGrowthSteps,
  'aml-forest-vs-single-tree': ForestVsSingleTree,

  // Lesson 03 – Dimensionality Reduction
  'aml-pca-correlated-data': PcaCorrelatedDataDemo,
  'aml-explained-variance': ExplainedVarianceDemo,
  'aml-tsne-umap-comparison': SwissRollExplorer,

  // Lesson 04 – Neural Network Fundamentals
  'aml-backprop-blame': BackpropBlameGame,
  'aml-forward-backward-pass': ForwardBackwardPass,
  'aml-activation-gallery': ActivationGallery,

  // Lesson 05 – Graph Neural Networks
  'aml-graph-message-passing': GnnMessagePassingLive,

  // Lesson 06 – Recurrent Neural Networks
  'aml-rnn-memory-highway': RnnMemoryHighway,

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
  "aml-graph-message-passing": "GNN message passing live — interactive graph with step-by-step aggregation and oversmoothing detection.",
  "aml-gan-forger-arena": "GAN forger arena — generator learns a target distribution with live loss curves and mode collapse detection.",
  "aml-loss-curves-outliers": "Loss curves under outliers — comparing MSE, MAE, and Huber regression fits on data with controllable outliers.",
  "aml-tree-growth-steps": "Tree growth steps — step-by-step visualization of recursive partitioning with Gini impurity at each split.",
  "aml-forest-vs-single-tree": "Forest vs single tree — comparing a deep overfitting tree against a random forest on noisy data with adjustable tree count.",
  "aml-forward-backward-pass": "Forward and backward pass — animated numerical values flowing through a 2-3-1 MLP with gradient magnitudes shown at each node.",
  "aml-activation-gallery": "Activation gallery — side-by-side plots of ReLU, Sigmoid, Tanh, Leaky ReLU, and Swish with their derivatives.",
};

export const AML_FIGURES: Record<string, { src: string; caption: string }> = {
  'aml-loss-outlier-comparison': {
    src: '/figures/aml-loss-outlier-comparison.svg',
    caption: 'Loss function comparison under outliers. MSE fit is dragged toward the three red outlier points; MAE and Huber fits stay anchored to the blue data majority. Bottom-right panel shows the penalty curves — MSE grows quadratically, MAE linearly, and Huber transitions smoothly between the two.',
  },
  'aml-tree-growth-steps': {
    src: '/figures/aml-tree-growth-steps.svg',
    caption: 'Decision tree recursive partitioning in four steps. Step 0: unsplit data with Gini 0.5. Steps 1-3: successive axis-aligned splits reduce impurity to zero, partitioning blue and amber classes into pure leaf regions.',
  },
  'aml-activation-gallery': {
    src: '/figures/aml-activation-gallery.svg',
    caption: 'Activation function gallery. Top row: function curves (solid) with derivative overlays (dashed). Bottom row: gradient-flow heatmaps through 20 layers.',
  },
  'aml-numeric-forward-backward': {
    src: '/figures/aml-numeric-forward-backward.svg',
    caption: 'Numeric forward and backward pass through a 2-3-1 MLP showing gradient magnitudes shrinking from output to input.',
  },
};
