'use client';

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

interface SimulationProps {
  id: string;
}

export const ADVANCED_DEEP_LEARNING_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
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
};
