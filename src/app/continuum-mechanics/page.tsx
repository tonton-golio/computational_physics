'use client';

import React from 'react';
import { MarkdownContent } from '@/components/content/MarkdownContent';

const content = `# Continuum Mechanics

## Introduction
In the macroscopic world, most materials that surround us e.g. solids and liquids can safely be assumed to exist as continua, that is, the materials completely fill the space they occupy and the underlying atomic structures can be neglected. This course offers a modern introduction to the physics of continuous matter with an emphasis on examples from natural occurring systems (e.g. in the physics of complex systems and the Earth sciences). Focus is equally on the underlying formalism of continuum mechanics and phenomenology. In the course you will become familiar with the mechanical behavior of materials ranging from viscous fluids to elastic solids.

A description of the deformation of solids is given, including the concepts of mechanical equilibrium, the stress and strain tensors, linear elasticity. A derivation is given of the Navier-Cauchy equation as well as examples from elastostatics and elastodynamics including elastic waves. A description of fluids in motion is given including the Euler equations, potential flow, Stokes' flow and the Navier-Stokes equation. Examples of numerical modeling of continous matter will be given. [from course description](https://kurser.ku.dk/course/nfyk10005u)

## 1D Stress-Strain Relationship

Explore Hooke's law and nonlinear elasticity using the Ramberg-Osgood model.

[[simulation stress-strain-sim]]

### Equations
- **Hooke's Law (Linear):** σ = E ε
- **Ramberg-Osgood (Nonlinear):** ε = σ/E + α (σ/E)^n

The Ramberg-Osgood model captures nonlinear elastic behavior, often used in materials science for metals exhibiting power-law hardening.
`;

export default function ContinuumMechanicsPage() {
  return (
    <div className="min-h-screen bg-[#0a0a15] text-white">
      <div className="container mx-auto px-4 py-8">
        <MarkdownContent content={content} />
      </div>
    </div>
  );
}