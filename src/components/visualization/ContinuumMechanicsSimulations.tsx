'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import StressStrainCurve from './continuum-mechanics/StressStrainCurve';
import StressTensor from './continuum-mechanics/StressTensor';
import ElasticWave from './continuum-mechanics/ElasticWave';
import StressTensorDiagram from './continuum-mechanics/StressTensorDiagram';
import HookesLawDiagram from './continuum-mechanics/HookesLawDiagram';
import DensityFluctuations from './continuum-mechanics/DensityFluctuations';
import AveragingVolume from './continuum-mechanics/AveragingVolume';
import DeformationGrid from './continuum-mechanics/DeformationGrid';
import PSWaveAnimation from './continuum-mechanics/PSWaveAnimation';
import ArchimedesStability from './continuum-mechanics/ArchimedesStability';
import BernoulliStreamline from './continuum-mechanics/BernoulliStreamline';
import BoundaryLayer from './continuum-mechanics/BoundaryLayer';
import PoiseuilleVsPowerLaw from './continuum-mechanics/PoiseuilleVsPowerLaw';
import DispersionRelation from './continuum-mechanics/DispersionRelation';
import StokesFlowDemo from './continuum-mechanics/StokesFlowDemo';
import GlacierCrossSection from './continuum-mechanics/GlacierCrossSection';
import FEMConvergence from './continuum-mechanics/FEMConvergence';
import UnifiedMap from './continuum-mechanics/UnifiedMap';
import type { SimulationComponentProps } from '@/shared/types/simulation';


// Stress-Strain Simulation (legacy, kept for backward compatibility)
function StressStrainSim({ }: SimulationComponentProps) {
  const [params, setParams] = useState({
    E: 200, // Young's modulus, GPa
    alpha: 0.002,
    n: 5,
    maxStrain: 0.05, // 5%
    numPoints: 100
  });
  // Bisection method to solve for sigma given epsilon
  const solveSigma = useCallback((epsilon: number, E: number, alpha: number, n: number): number => {
    if (epsilon === 0) return 0;
    let low = 0;
    let high = E * epsilon * 10; // upper bound
    let mid = 0;
    for (let i = 0; i < 100; i++) {
      mid = (low + high) / 2;
      const elastic = mid / E;
      const plastic = alpha * Math.pow(mid / E, n);
      const calcEpsilon = elastic + plastic;
      if (calcEpsilon < epsilon) {
        low = mid;
      } else {
        high = mid;
      }
    }
    return mid;
  }, []);

  const generateData = useMemo(() => {
    const { E, alpha, n, maxStrain, numPoints } = params;
    const epsilonValues: number[] = [];
    const linearSigma: number[] = [];
    const nonlinearSigma: number[] = [];

    for (let i = 0; i <= numPoints; i++) {
      const epsilon = (i / numPoints) * maxStrain;
      epsilonValues.push(epsilon);
      linearSigma.push(E * epsilon);
      nonlinearSigma.push(solveSigma(epsilon, E, alpha, n));
    }

    const data = [
      {
        type: 'scatter',
        mode: 'lines',
        x: epsilonValues,
        y: linearSigma,
        name: 'Linear (Hooke\'s Law)',
        line: { color: '#3b82f6' }
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: epsilonValues,
        y: nonlinearSigma,
        name: 'Nonlinear (Ramberg-Osgood)',
        line: { color: '#ef4444' }
      }
    ];

    const layout = {
      title: { text: 'Stress-Strain Curve' },
      xaxis: { title: { text: 'Strain (\u03b5)' } },
      yaxis: { title: { text: 'Stress (\u03c3) [GPa]' } },
    };

    return { data, layout };
  }, [params, solveSigma]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Stress-Strain Simulation</h3>
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-[var(--text-strong)]">Young&apos;s Modulus (E) [GPa]: {params.E}</label>
          <Slider
            min={50}
            max={500}
            step={10}
            value={[params.E]}
            onValueChange={([v]) => setParams(p => ({ ...p, E: v }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Hardening Coefficient (&alpha;): {params.alpha.toFixed(3)}</label>
          <Slider
            min={0.001}
            max={0.01}
            step={0.001}
            value={[params.alpha]}
            onValueChange={([v]) => setParams(p => ({ ...p, alpha: v }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Hardening Exponent (n): {params.n}</label>
          <Slider
            min={1}
            max={10}
            step={0.5}
            value={[params.n]}
            onValueChange={([v]) => setParams(p => ({ ...p, n: v }))}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-[var(--text-strong)]">Max Strain: {params.maxStrain.toFixed(3)}</label>
          <Slider
            min={0.01}
            max={0.1}
            step={0.005}
            value={[params.maxStrain]}
            onValueChange={([v]) => setParams(p => ({ ...p, maxStrain: v }))}
            className="w-full"
          />
        </div>
      </div>
      <CanvasChart
        data={generateData.data}
        layout={generateData.layout}
        style={{ width: '100%', height: 500 }}
      />
      <div className="mt-4 text-sm text-[var(--text-muted)]">
        <p>The linear curve follows Hooke&apos;s law: &sigma; = E&epsilon;</p>
        <p>The nonlinear curve uses the Ramberg-Osgood model: &epsilon; = &sigma;/E + &alpha;(&sigma;/E)^n</p>
      </div>
    </div>
  );
}

// Gaussian elimination with partial pivoting
function gaussElim(A: number[][], b: number[]): number[] {
  const n = A.length;
  const a = A.map(row => [...row]);
  const bb = [...b];
  for (let p = 0; p < n; p++) {
    let max = p;
    for (let i = p + 1; i < n; i++) {
      if (Math.abs(a[i][p]) > Math.abs(a[max][p])) max = i;
    }
    [a[p], a[max]] = [a[max], a[p]];
    [bb[p], bb[max]] = [bb[max], bb[p]];
    for (let i = p + 1; i < n; i++) {
      const alpha = a[i][p] / a[p][p];
      bb[i] -= alpha * bb[p];
      for (let j = p; j < n; j++) a[i][j] -= alpha * a[p][j];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += a[i][j] * x[j];
    x[i] = (bb[i] - sum) / a[i][i];
  }
  return x;
}

// FEM 1D Bar Simulation
function FEM1DBarSim({ }: SimulationComponentProps) {
  const [ne, setNe] = useState(3);
  const [showBasis, setShowBasis] = useState(true);

  // Problem: 1D bar, fixed at x=0, free at x=1, uniform distributed load q
  // EA = L = q = 1  =>  exact: u(x) = x - x^2/2

  const fem = useMemo(() => {
    const n = ne + 1;
    const le = 1 / ne;
    const kf = 1 / le;

    const K = Array.from({ length: n }, () => Array(n).fill(0));
    for (let e = 0; e < ne; e++) {
      K[e][e] += kf;
      K[e][e + 1] -= kf;
      K[e + 1][e] -= kf;
      K[e + 1][e + 1] += kf;
    }

    const f = Array(n).fill(0);
    for (let e = 0; e < ne; e++) {
      f[e] += le / 2;
      f[e + 1] += le / 2;
    }

    // BC: u(0) = 0
    for (let i = 0; i < n; i++) { K[0][i] = 0; K[i][0] = 0; }
    K[0][0] = 1;
    f[0] = 0;

    const U = gaussElim(K, f);
    const X = Array.from({ length: n }, (_, i) => i * le);
    return { U, X };
  }, [ne]);

  const data = useMemo(() => {
    // Exact solution at fine resolution
    const exactX = Array.from({ length: 200 }, (_, i) => i / 199);
    const exactU = exactX.map(x => x - x * x / 2);

    const traces: {
      x: number[]; y: number[];
      mode?: string; fill?: string; fillcolor?: string;
      line?: { color: string; width?: number; dash?: string };
      marker?: { color: string; size?: number };
      name?: string; showlegend?: boolean;
    }[] = [];

    // Basis functions (hat functions scaled by nodal displacement)
    if (showBasis) {
      const { U, X } = fem;
      for (let i = 1; i < U.length; i++) {
        if (Math.abs(U[i]) < 1e-14) continue;
        const hx: number[] = [];
        const hy: number[] = [];
        if (i > 0) { hx.push(X[i - 1]); hy.push(0); }
        hx.push(X[i]); hy.push(U[i]);
        if (i < U.length - 1) { hx.push(X[i + 1]); hy.push(0); }
        traces.push({
          x: hx, y: hy,
          mode: 'lines',
          fill: 'tozeroy',
          fillcolor: 'rgba(245, 158, 11, 0.12)',
          line: { color: 'rgba(245, 158, 11, 0.45)', width: 1 },
          showlegend: false,
        });
      }
    }

    // Exact solution
    traces.push({
      x: exactX, y: exactU,
      mode: 'lines',
      line: { color: '#8fa3c4', width: 2, dash: 'dash' },
      name: 'Exact',
    });

    // FEM solution
    traces.push({
      x: fem.X, y: fem.U,
      mode: 'lines+markers',
      line: { color: '#3b82f6', width: 2 },
      marker: { color: '#3b82f6', size: 5 },
      name: 'FEM',
    });

    return traces;
  }, [fem, showBasis]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-2 text-[var(--text-strong)]">
        1D Bar Under Distributed Load
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        Fixed at left, uniform body force. The exact displacement is a parabola &mdash;
        linear elements approximate it with straight segments.
      </p>
      <div className="flex items-center gap-6 mb-4">
        <div className="flex-1 max-w-xs">
          <label className="text-[var(--text-strong)] text-sm">Elements: {ne}</label>
          <Slider min={1} max={16} value={[ne]} onValueChange={([v]) => setNe(v)} className="w-full" />
        </div>
        <label className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showBasis}
            onChange={e => setShowBasis(e.target.checked)}
            className="rounded"
          />
          Show basis functions
        </label>
      </div>
      <CanvasChart
        data={data}
        layout={{
          xaxis: { title: { text: 'Position x' } },
          yaxis: { title: { text: 'Displacement u(x)' } },
        }}
        style={{ width: '100%', height: 400 }}
      />
    </div>
  );
}

export const CONTINUUM_MECHANICS_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'stress-strain-sim': StressStrainSim,
  'fem-1d-bar-sim': FEM1DBarSim,
  'stress-strain-curve': StressStrainCurve,
  'mohr-circle': StressTensor,
  'elastic-wave': ElasticWave,
  'stress-tensor-diagram': StressTensorDiagram,
  'hookes-law-diagram': HookesLawDiagram,
  'density-fluctuations': DensityFluctuations,
  'averaging-volume': AveragingVolume,
  'deformation-grid': DeformationGrid,
  'ps-wave-animation': PSWaveAnimation,
  'archimedes-stability': ArchimedesStability,
  'bernoulli-streamline': BernoulliStreamline,
  'boundary-layer': BoundaryLayer,
  'poiseuille-vs-power-law': PoiseuilleVsPowerLaw,
  'dispersion-relation': DispersionRelation,
  'stokes-flow-demo': StokesFlowDemo,
  'glacier-cross-section': GlacierCrossSection,
  'fem-convergence': FEMConvergence,
  'unified-map': UnifiedMap,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const CONTINUUM_DESCRIPTIONS: Record<string, string> = {
  "stress-strain-sim": "Stress-strain simulation — comparing linear (Hooke's law) and nonlinear (Ramberg–Osgood) material responses under loading.",
  "fem-1d-bar-sim": "1D finite element method — solving for displacement in a bar under distributed load using linear basis functions.",
  "stress-strain-curve": "Stress-strain curve visualization — exploring elastic and plastic deformation regimes in materials under tensile loading.",
  "mohr-circle": "Mohr's circle — graphical representation of the stress state at a point, showing principal stresses and maximum shear.",
  "elastic-wave": "Elastic wave propagation — visualizing P-waves and S-waves traveling through an elastic medium.",
  "stress-tensor-diagram": "Stress tensor diagram — the components of the Cauchy stress tensor acting on an infinitesimal element.",
  "hookes-law-diagram": "Hooke's law — the linear relationship between stress and strain in the elastic regime of material deformation.",
  "density-fluctuations": "Density fluctuation analysis — examining spatial variations in material density and their statistical properties.",
  "averaging-volume": "Averaging volume convergence — showing how density fluctuations decrease as 1/sqrt(N) with increasing averaging volume, validating the continuum approximation.",
  "deformation-grid": "2D deformation grid — visualising how different strain states (pure shear, simple shear, uniaxial, biaxial) transform a regular grid.",
  "ps-wave-animation": "P-wave vs S-wave animation — contrasting longitudinal (compressional) and transverse (shear) particle displacements in elastic wave propagation.",
  "archimedes-stability": "Archimedes' principle and floating stability — showing buoyancy, draft, and metacentric height for a rectangular floating body.",
  "bernoulli-streamline": "Bernoulli streamline — flow through a converging-diverging channel showing velocity increase and pressure drop at the throat (Venturi effect).",
  "boundary-layer": "Boundary layer profiles — Blasius velocity profiles at different stations along a flat plate, showing boundary layer growth with downstream distance.",
  "poiseuille-vs-power-law": "Poiseuille vs power-law flow — comparing Newtonian parabolic and power-law velocity profiles in pipe flow, including glacier rheology (Glen's law).",
  "dispersion-relation": "Gravity-capillary dispersion relation — plotting omega vs k for gravity and capillary waves, with phase and group velocities and water depth dependence.",
  "stokes-flow-demo": "Stokes creeping flow — streamlines around a sphere at low Reynolds number with fore-aft symmetry, and Stokes drag scaling linearly with velocity.",
  "glacier-cross-section": "Glacier cross-section — ice flow velocity and shear stress profiles through a glacier slab using Glen's power-law rheology.",
  "fem-convergence": "FEM convergence study — L2 error vs number of elements for the 1D bar problem, confirming O(h^2) convergence for linear elements.",
  "unified-map": "Continuum mechanics concept map — a node-link diagram showing the logical connections between the main topics from continuum approximation to FEM.",
};
