'use client';

import React from 'react';
import PCAVisualization from './advanced-deep-learning/PCAVisualization';
import ActivationFunctions from './advanced-deep-learning/ActivationFunctions';
import ConvolutionDemo from './advanced-deep-learning/ConvolutionDemo';
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
};
