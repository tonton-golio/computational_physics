import { MarkdownContent } from '@/components/content/MarkdownContent';
import { FEMSim } from '@/components/visualization/dynamical-models/FEMSim';

const content = `# FEM 1D Bar Element

## Basics

Finite Element Method for a 1D axial bar:

- **Geometry**: Length L, uniform cross-section A
- **Material**: Young's modulus E
- **Loading**: Fixed at left (u(0)=0), axial force P at right
- **Discretization**: ne linear elements, n=ne+1 nodes
- **Element stiffness**:
  $$ k^{(e)} = \\frac{AE}{l_e} \\begin{bmatrix} 1 & -1 \\\\ -1 & 1 \\end{bmatrix} $$
  where $l_e = L/ne$

- **Assembly**: Global K (n x n) by summing contributions
- **Boundary conditions**: Eliminate u_1=0, apply f_n = P
- **Solution**: Solve reduced K_red u_red = f_red
- **Post-process**: Interpolate u(x), plot deformed shape

## Analytical Solution
$u(x) = \\frac{P}{AE} x$

## Implementation
Interactive simulation below.

## Visualization
- Original mesh (gray)
- Deformed mesh (blue, exaggerated)
- Displacement u(x) plot
- "Contours": color elements by average displacement or strain (constant here)
`;

export default function FEMPage() {
  return (
    <div className="min-h-screen bg-[#0a0a15] text-white">
      <div className="container mx-auto px-4 py-8">
        <MarkdownContent content={content} />
        <div className="mt-12">
          <h2 className="text-2xl font-bold mb-6">Interactive FEM Simulation</h2>
          <FEMSim />
        </div>
      </div>
    </div>
  );
}