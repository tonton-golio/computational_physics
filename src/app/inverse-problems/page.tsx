'use client';

import React from 'react';
import { MarkdownContent } from '@/components/content/MarkdownContent';

const content = `# Inverse Problems

Inverse problems involve determining the parameters of a system from observed data. This is common in many scientific fields, including physics, engineering, and biology.

## Parameter Estimation in Lotka-Volterra Model

The Lotka-Volterra equations model predator-prey dynamics:

$$
\\frac{dx}{dt} = \\alpha x - \\beta x y
$$

$$
\\frac{dy}{dt} = -\\gamma y + \\delta x y
$$

Where:
- x: prey population
- y: predator population
- α: prey growth rate
- β: predation rate
- γ: predator death rate
- δ: predator efficiency

In this simulation, we have "observed" data from a system with known true parameters. Adjust the parameters to fit the data and minimize residuals.

[[simulation lotka-volterra-estimation]]

### Inverse Problem Challenge
Given the observed population dynamics, estimate the underlying parameters α, β, γ, δ. This is an example of parameter estimation in nonlinear ODEs, which is fundamental to many inverse problems in computational physics and biology.
`;

export default function InverseProblemsPage() {
  return (
    <div className="min-h-screen bg-[#0a0a15] text-white">
      <div className="container mx-auto px-4 py-8">
        <MarkdownContent content={content} />
      </div>
    </div>
  );
}