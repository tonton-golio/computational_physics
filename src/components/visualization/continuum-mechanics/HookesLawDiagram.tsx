"use client";

import { useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Static stress-strain curve highlighting the linear Hooke's-law regime,
 * yield point, and plastic region.  No sliders — just a clean overview figure.
 */
export default function HookesLawDiagram({}: SimulationComponentProps) {
  const plotData = useMemo(() => {
    const E = 200;          // Young's modulus (GPa)
    const sigmaY = 0.30;    // Yield stress (GPa)
    const n = 8;            // Ramberg-Osgood exponent
    const alpha = 0.002;
    const maxStrain = 0.06;
    const numPoints = 300;

    function solveSigma(targetEps: number): number {
      if (targetEps <= 0) return 0;
      let lo = 0;
      let hi = E * targetEps * 2;
      for (let i = 0; i < 80; i++) {
        const mid = (lo + hi) / 2;
        const eps = mid / E + alpha * Math.pow(mid / sigmaY, n);
        if (eps < targetEps) lo = mid;
        else hi = mid;
      }
      return (lo + hi) / 2;
    }

    const eps: number[] = [];
    const linear: number[] = [];
    const nonlinear: number[] = [];
    for (let i = 0; i <= numPoints; i++) {
      const e = (i / numPoints) * maxStrain;
      eps.push(e);
      linear.push(E * e);
      nonlinear.push(solveSigma(e));
    }

    // Yield point (0.2 % offset)
    const yieldEps = sigmaY / E + alpha;

    // Elastic region shading (vertical fill from 0 to yield strain)
    const elasticEps: number[] = [];
    const elasticSig: number[] = [];
    for (let i = 0; i <= 100; i++) {
      const e = (i / 100) * yieldEps;
      elasticEps.push(e);
      elasticSig.push(solveSigma(e));
    }

    return {
      data: [
        // Shaded elastic region
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: elasticEps,
          y: elasticSig,
          fill: 'tozeroy' as const,
          fillcolor: 'rgba(59, 130, 246, 0.10)',
          line: { color: 'transparent', width: 0 },
          name: 'Elastic region',
          showlegend: true,
        },
        // Linear Hooke's law
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: eps.filter((_, i) => eps[i] <= yieldEps * 1.3),
          y: linear.filter((_, i) => eps[i] <= yieldEps * 1.3),
          name: "Hooke's Law  \u03c3 = E\u03b5",
          line: { color: '#3b82f6', width: 2, dash: 'dash' as const },
        },
        // Full nonlinear curve
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: eps,
          y: nonlinear,
          name: 'Material response',
          line: { color: '#ef4444', width: 2.5 },
        },
        // Yield point marker
        {
          type: 'scatter' as const,
          mode: 'markers' as const,
          x: [yieldEps],
          y: [sigmaY],
          name: 'Yield point',
          marker: { color: '#f59e0b', size: 9, symbol: 'diamond' },
        },
      ],
      layout: {
        xaxis: { title: { text: 'Strain \u03b5' } },
        yaxis: { title: { text: 'Stress \u03c3 [GPa]' } },
        height: 400,
        legend: { bgcolor: 'rgba(0,0,0,0)' },
        margin: { t: 20, b: 55, l: 65, r: 20 },
        annotations: [
          {
            x: yieldEps / 2,
            y: sigmaY * 0.45,
            text: 'Linear<br>elastic',
            showarrow: false,
            font: { color: '#93c5fd', size: 12 },
          },
          {
            x: maxStrain * 0.7,
            y: nonlinear[Math.round(numPoints * 0.7)] * 0.95,
            text: 'Plastic<br>hardening',
            showarrow: false,
            font: { color: '#fca5a5', size: 12 },
          },
        ],
      },
    };
  }, []);

  return (
    <SimulationPanel title="Hooke's Law & the Stress-Strain Curve" caption="In the linear elastic region (shaded blue) stress is proportional to strain: \u03c3 = E\u03b5. Beyond the yield point the material deforms plastically and the relationship becomes nonlinear.">
      <SimulationMain>
        <CanvasChart
          data={plotData.data}
          layout={plotData.layout}
          style={{ width: '100%', height: 400 }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
