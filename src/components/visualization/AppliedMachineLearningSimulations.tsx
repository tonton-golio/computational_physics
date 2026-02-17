'use client';

import React from 'react';
import {
  ExplainedVarianceDemo,
  GraphAdjacencyDemo,
  GraphConvolutionIntuitionDemo,
  GraphMessagePassingDemo,
  LossFunctionsDemo,
  LossLandscapeDemo,
  PcaCorrelatedDataDemo,
  TreeEnsembleXorDemo,
  TreeSplitImpurityDemo,
  TsneUmapComparisonDemo,
  ValidationSplitDemo,
} from './applied-machine-learning/AppliedMachineLearningDemos';

interface SimulationProps {
  id: string;
}

export const APPLIED_MACHINE_LEARNING_SIMULATIONS: Record<string, React.ComponentType<SimulationProps>> = {
  'aml-loss-functions': LossFunctionsDemo,
  'aml-loss-landscape': LossLandscapeDemo,
  'aml-validation-split': ValidationSplitDemo,
  'aml-tree-split-impurity': TreeSplitImpurityDemo,
  'aml-tree-ensemble-xor': TreeEnsembleXorDemo,
  'aml-pca-correlated-data': PcaCorrelatedDataDemo,
  'aml-explained-variance': ExplainedVarianceDemo,
  'aml-tsne-umap-comparison': TsneUmapComparisonDemo,
  'aml-graph-convolution-intuition': GraphConvolutionIntuitionDemo,
  'aml-graph-adjacency-demo': GraphAdjacencyDemo,
  'aml-graph-message-passing': GraphMessagePassingDemo,
};
