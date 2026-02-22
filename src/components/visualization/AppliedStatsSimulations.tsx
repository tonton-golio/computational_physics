"use client";

import React, { useState, useMemo } from 'react';
import * as math from 'mathjs';
import { CanvasChart } from '@/components/ui/canvas-chart';
import RegressionAnalysis from './applied-statistics/RegressionAnalysis';
import DataExploration from './applied-statistics/DataExploration';
import BayesianInference from './applied-statistics/BayesianInference';
import ConfidenceIntervals from './applied-statistics/ConfidenceIntervals';
import CorrelationAnalysis from './applied-statistics/CorrelationAnalysis';
import BootstrapResampling from './applied-statistics/BootstrapResampling';
import WhichAverage from './applied-statistics/WhichAverage';
import WeightedMean from './applied-statistics/WeightedMean';
import LikelihoodSurface from './applied-statistics/LikelihoodSurface';
import PoissonToGaussian from './applied-statistics/PoissonToGaussian';
import AcceptReject from './applied-statistics/AcceptReject';
import MonteCarloIntegrationStats from './applied-statistics/MonteCarloIntegrationStats';
import ResidualPattern from './applied-statistics/ResidualPattern';
import VarianceDecomposition from './applied-statistics/VarianceDecomposition';
import InteractionSurface from './applied-statistics/InteractionSurface';
import RandomizationVsBlocking from './applied-statistics/RandomizationVsBlocking';
import ShrinkagePlot from './applied-statistics/ShrinkagePlot';
import SpaghettiTrajectory from './applied-statistics/SpaghettiTrajectory';
import MissingnessMechanisms from './applied-statistics/MissingnessMechanisms';
import PriorInfluence from './applied-statistics/PriorInfluence';
import ProfileLikelihood from './applied-statistics/ProfileLikelihood';
import OverfittingCarousel from './applied-statistics/OverfittingCarousel';
import ROCLive from './applied-statistics/ROCLive';
import { Slider } from '@/components/ui/slider';
import type { SimulationComponentProps } from '@/shared/types/simulation';


// Central Limit Theorem Simulation
function CentralLimitTheoremSim() {
  const [sampleSize, setSampleSize] = useState(10);
  const [numSamples, setNumSamples] = useState(500);
  const data = useMemo<number[]>(() => {
    const samples = [];
    for (let i = 0; i < numSamples; i++) {
      const sample = [];
      for (let j = 0; j < sampleSize; j++) {
        sample.push(math.random(0, 1)); // uniform [0,1]
      }
      const mean = math.mean(sample);
      samples.push(mean);
    }
    return samples;
  }, [sampleSize, numSamples]);

  const histData = data.length > 0 ? data : [];
  const hist = histData.length > 0 ? {
    x: histData,
    type: 'histogram',
    nbinsx: 50,
    name: 'Sample Means'
  } : null;

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Central Limit Theorem</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-[var(--text-strong)]">Sample Size: {sampleSize}</label>
          <Slider
            min={1}
            max={100}
            step={1}
            value={[sampleSize]}
            onValueChange={([v]) => setSampleSize(v)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Number of Samples: {numSamples}</label>
          <Slider
            min={100}
            max={1000}
            step={50}
            value={[numSamples]}
            onValueChange={([v]) => setNumSamples(v)}
            className="w-full"
          />
        </div>
      </div>
      <CanvasChart
        data={hist ? [hist as any] : []}
        layout={{
          title: { text: `Distribution of Sample Means (n=${sampleSize})` },
          xaxis: { title: { text: 'Sample Mean' } },
          yaxis: { title: { text: 'Frequency' } },
          height: 400,
        }}
      />
    </div>
  );
}

// Hypothesis Testing Simulation
function HypothesisTestingSim() {
  const [group1, setGroup1] = useState('1,2,3,4,5');
  const [group2, setGroup2] = useState('2,3,4,5,6');
  const { tStat, meanDiff } = useMemo(() => {
    try {
      const g1 = group1.split(',').map(Number);
      const g2 = group2.split(',').map(Number);
      if (g1.some(isNaN) || g2.some(isNaN)) return { tStat: 0, meanDiff: 0 };

      const mean1 = math.mean(g1);
      const mean2 = math.mean(g2);
      const var1 = math.number(math.variance(g1));
      const var2 = math.number(math.variance(g2));
      const n1 = g1.length;
      const n2 = g2.length;

      const pooledVar = ((n1 - 1) * Number(var1) + (n2 - 1) * Number(var2)) / (n1 + n2 - 2);
      const se = Math.sqrt(pooledVar * (1/n1 + 1/n2));
      const t = (mean1 - mean2) / se;

      return { tStat: t, meanDiff: mean1 - mean2 };
    } catch (e) {
      console.error(e);
      return { tStat: 0, meanDiff: 0 };
    }
  }, [group1, group2]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Hypothesis Testing (Two-Sample t-Test)</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-[var(--text-strong)]">Group 1 (comma-separated):</label>
          <input
            type="text"
            value={group1}
            onChange={(e) => setGroup1(e.target.value)}
            className="w-full p-2 bg-[var(--surface-2)] border border-[var(--border-strong)] rounded text-[var(--text-strong)]"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Group 2 (comma-separated):</label>
          <input
            type="text"
            value={group2}
            onChange={(e) => setGroup2(e.target.value)}
            className="w-full p-2 bg-[var(--surface-2)] border border-[var(--border-strong)] rounded text-[var(--text-strong)]"
          />
        </div>
      </div>
      <div className="mb-4 text-[var(--text-muted)]">
        <p>t-statistic: {tStat.toFixed(3)}</p>
        <p>Mean difference: {meanDiff.toFixed(3)}</p>
        <p>Note: For significance, |t| {'>'} 2 suggests difference (approx.)</p>
      </div>
    </div>
  );
}

export const APPLIED_STATS_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'applied-stats-sim-1': DataExploration,
  'applied-stats-sim-2': RegressionAnalysis,
  'applied-stats-sim-3': CentralLimitTheoremSim,
  'applied-stats-sim-4': HypothesisTestingSim,
  'applied-stats-sim-5': BayesianInference,
  'applied-stats-sim-6': ConfidenceIntervals,
  'applied-stats-sim-7': CorrelationAnalysis,
  'applied-stats-sim-8': BootstrapResampling,
  'which-average': WhichAverage,
  'weighted-mean': WeightedMean,
  'likelihood-surface': LikelihoodSurface,
  'poisson-to-gaussian': PoissonToGaussian,
  'accept-reject': AcceptReject,
  'monte-carlo-integration-stats': MonteCarloIntegrationStats,
  'residual-pattern': ResidualPattern,
  'variance-decomposition': VarianceDecomposition,
  'interaction-surface': InteractionSurface,
  'randomization-vs-blocking': RandomizationVsBlocking,
  'shrinkage-plot': ShrinkagePlot,
  'spaghetti-trajectory': SpaghettiTrajectory,
  'missingness-mechanisms': MissingnessMechanisms,
  'prior-influence': PriorInfluence,
  'profile-likelihood': ProfileLikelihood,
  'overfitting-carousel': OverfittingCarousel,
  'roc-live': ROCLive,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const STATS_DESCRIPTIONS: Record<string, string> = {
  "applied-stats-sim-1": "Statistical data exploration — interactive summary statistics and distribution visualization for sample data.",
  "applied-stats-sim-2": "Regression analysis — fitting linear and polynomial models to data and evaluating goodness of fit.",
  "applied-stats-sim-3": "Central Limit Theorem — the distribution of sample means approaches a Gaussian as sample size increases.",
  "applied-stats-sim-4": "Two-sample t-test — testing whether two groups have significantly different means using pooled variance.",
  "applied-stats-sim-5": "Bayesian inference — updating prior beliefs with observed data to obtain posterior distributions.",
  "applied-stats-sim-6": "Confidence intervals — visualizing the range of plausible parameter values at a given significance level.",
  "applied-stats-sim-7": "Correlation analysis — measuring the strength and direction of linear relationships between variables.",
  "applied-stats-sim-8": "Bootstrap resampling — estimating sampling distributions by repeatedly drawing with replacement from the data.",
  "which-average": "Compare mean, median, and mode on a skewed distribution — see how each measure of central tendency responds to skewness.",
  "weighted-mean": "Weighted mean visualization — measurements with smaller uncertainties get more weight, pulling the average toward precise readings.",
  "likelihood-surface": "2D likelihood surface for Gaussian parameters — add or remove data points and watch the likelihood peak shift.",
  "poisson-to-gaussian": "Poisson-to-Gaussian convergence — animate the Poisson distribution as lambda increases, showing it approaching a bell curve.",
  "accept-reject": "Accept-reject sampling — visualize proposal distribution, target density, and accepted vs rejected random points.",
  "monte-carlo-integration-stats": "Monte Carlo integration — random points estimate an integral, with running convergence toward the true value.",
  "residual-pattern": "Residual diagnostics — compare residual patterns from a well-specified model vs a misspecified one.",
  "variance-decomposition": "ANOVA variance decomposition — adjust between-group and within-group spread to see F-statistic and SS breakdown.",
  "interaction-surface": "Factorial experiment interaction — two factors with adjustable main effects and interaction; parallel lines mean no interaction.",
  "randomization-vs-blocking": "Randomization vs blocking — side-by-side comparison showing how blocking removes nuisance variability.",
  "shrinkage-plot": "Bayesian shrinkage — group means pulled toward the grand mean, with stronger priors producing more shrinkage.",
  "spaghetti-trajectory": "Spaghetti plot of random trajectories — individual paths, mean trajectory, and confidence bands.",
  "missingness-mechanisms": "Missing data mechanisms — illustrate MCAR, MAR, and MNAR patterns in a scatter plot.",
  "prior-influence": "Prior influence on the posterior — slider for prior precision shows how the posterior interpolates between prior and likelihood.",
  "profile-likelihood": "Profile likelihood for a single parameter — confidence intervals derived from the likelihood ratio threshold.",
  "overfitting-carousel": "Overfitting demo — polynomial degree slider shows training error decreasing while test error rises.",
  "roc-live": "Interactive ROC curve — adjust the classification threshold and class separation to explore TPR, FPR, and AUC.",
};
