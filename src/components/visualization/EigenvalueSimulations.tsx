"use client";

import dynamic from 'next/dynamic';
import type { SimulationComponentProps } from '@/shared/types/simulation';

const EigenTransformation = dynamic(
  () => import('./eigenvalue/EigenTransformation'),
  { ssr: false }
);

const PowerMethodAnimation = dynamic(
  () => import('./eigenvalue/PowerMethodAnimation'),
  { ssr: false }
);

const HermitianDemo = dynamic(
  () => import('./eigenvalue/HermitianDemo'),
  { ssr: false }
);

const InverseIterationDemo = dynamic(
  () => import('./eigenvalue/InverseIterationDemo'),
  { ssr: false }
);

const GershgorinCircles = dynamic(
  () => import('./eigenvalue/GershgorinCircles'),
  { ssr: false }
);

const ConvergenceComparison = dynamic(
  () => import('./eigenvalue/ConvergenceComparison'),
  { ssr: false }
);

const QRAlgorithmAnimation = dynamic(
  () => import('./eigenvalue/QRAlgorithmAnimation'),
  { ssr: false }
);

const RayleighQuotientSurface = dynamic(
  () => import('./eigenvalue/RayleighQuotientSurface'),
  { ssr: false }
);

const MatrixExponentialSimulation = dynamic(
  () => import('./eigenvalue/MatrixExponentialSimulation'),
  { ssr: false }
);

const CharacteristicPolynomial = dynamic(
  () => import('./eigenvalue/CharacteristicPolynomial'),
  { ssr: false }
);

export const EIGENVALUE_SIMULATIONS: Record<string, React.ComponentType<SimulationComponentProps>> = {
  'eigen-transformation': EigenTransformation,
  'power-method-animation': PowerMethodAnimation,
  'hermitian-demo': HermitianDemo,
  'inverse-iteration': InverseIterationDemo,
  'gershgorin-circles': GershgorinCircles,
  'convergence-comparison': ConvergenceComparison,
  'qr-algorithm-animation': QRAlgorithmAnimation,
  'rayleigh-convergence': RayleighQuotientSurface,
  'matrix-exponential': MatrixExponentialSimulation,
  'characteristic-polynomial': CharacteristicPolynomial,
};

// ============ CO-LOCATED DESCRIPTIONS ============

export const EIGEN_DESCRIPTIONS: Record<string, string> = {
  "eigen-transformation": "Eigenvalue transformation — eigenvectors remain on their span under the matrix, scaling by their eigenvalue.",
  "power-method-animation": "Power method — iteratively multiplying by a matrix to converge on the dominant eigenvector.",
  "hermitian-demo": "Hermitian matrix properties — real eigenvalues and orthogonal eigenvectors from symmetric matrices.",
  "inverse-iteration": "Inverse iteration — shift-and-invert to converge on the eigenvalue nearest a chosen shift.",
  "gershgorin-circles": "Gershgorin circles — discs in the complex plane guaranteed to contain all eigenvalues of a matrix.",
  "convergence-comparison": "Eigenvalue algorithm convergence — comparing power method, inverse iteration, and Rayleigh quotient rates.",
  "qr-algorithm-animation": "QR algorithm — iterative QR factorization driving a matrix toward upper triangular (Schur) form.",
  "rayleigh-convergence": "Rayleigh quotient surface — stationary points of the Rayleigh quotient correspond to eigenvectors.",
  "matrix-exponential": "Matrix exponential — comparing the linear transformation A with the exponential map e^A on the unit circle.",
  "characteristic-polynomial": "Characteristic polynomial — the polynomial det(A - λI) whose roots are the eigenvalues of A.",
};

export const EIGEN_FIGURES: Record<string, { src: string; caption: string }> = {};
