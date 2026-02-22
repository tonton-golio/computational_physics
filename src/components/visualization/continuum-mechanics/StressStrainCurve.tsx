"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function StressStrainCurve({}: SimulationComponentProps) {
  const [youngsMod, setYoungsMod] = useState(200); // GPa
  const [yieldStress, setYieldStress] = useState(0.25); // GPa
  const [hardeningExp, setHardeningExp] = useState(5);
  const [maxStrain, setMaxStrain] = useState(0.05);

  const plotData = useMemo(() => {
    const numPoints = 200;
    const E = youngsMod;
    const sigmaY = yieldStress;
    const n = hardeningExp;

    // alpha for Ramberg-Osgood: epsilon = sigma/E + alpha*(sigma/sigmaY)^n * (sigmaY/E)
    // Standard form: epsilon = sigma/E + (sigmaY/E) * (sigma/sigmaY)^n
    // We use: epsilon = sigma/E + 0.002 * (sigma/sigmaY)^n  (0.2% offset convention)
    const alpha = 0.002;

    const epsilonValues: number[] = [];
    const linearSigma: number[] = [];
    const nonlinearSigma: number[] = [];

    // Solve for sigma given epsilon using bisection (Ramberg-Osgood is epsilon(sigma))
    function solveSigma(targetEps: number): number {
      if (targetEps <= 0) return 0;
      let lo = 0;
      let hi = E * targetEps * 2;
      for (let iter = 0; iter < 80; iter++) {
        const mid = (lo + hi) / 2;
        const eps = mid / E + alpha * Math.pow(mid / sigmaY, n);
        if (eps < targetEps) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      return (lo + hi) / 2;
    }

    for (let i = 0; i <= numPoints; i++) {
      const eps = (i / numPoints) * maxStrain;
      epsilonValues.push(eps);
      linearSigma.push(E * eps);
      nonlinearSigma.push(solveSigma(eps));
    }

    // Yield point marker (0.2% offset line intersects nonlinear curve)
    const yieldEps = sigmaY / E + alpha;
    const yieldSig = sigmaY;

    // 0.2% offset line for reference
    const offsetLineEps: number[] = [];
    const offsetLineSig: number[] = [];
    for (let i = 0; i <= 50; i++) {
      const sig = (i / 50) * sigmaY * 1.3;
      offsetLineEps.push(sig / E + alpha);
      offsetLineSig.push(sig);
    }

    return {
      data: [
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: epsilonValues,
          y: linearSigma,
          name: "Hooke's Law (linear)",
          line: { color: '#3b82f6', width: 2 },
        },
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: epsilonValues,
          y: nonlinearSigma,
          name: 'Ramberg-Osgood (nonlinear)',
          line: { color: '#ef4444', width: 2.5 },
        },
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x: offsetLineEps,
          y: offsetLineSig,
          name: '0.2% offset line',
          line: { color: '#9ca3af', width: 1, dash: 'dash' as const },
        },
        {
          type: 'scatter' as const,
          mode: 'markers' as const,
          x: [yieldEps],
          y: [yieldSig],
          name: `Yield point (${sigmaY.toFixed(2)} GPa)`,
          marker: { color: '#f59e0b', size: 10, symbol: 'diamond' },
        },
      ],
      layout: ({
        title: { text: 'Stress-Strain Curve' },
        xaxis: {
          title: { text: 'Strain \u03b5' },
        },
        yaxis: {
          title: { text: 'Stress \u03c3 [GPa]' },
        },
        height: 500,
        legend: {
          bgcolor: 'rgba(0,0,0,0)',
        },
        margin: { t: 50, b: 60, l: 70, r: 30 },
      }),
    };
  }, [youngsMod, yieldStress, hardeningExp, maxStrain]);

  return (
    <SimulationPanel title="Interactive Stress-Strain Curve" caption="Compare linear elastic response (Hooke's law) with nonlinear Ramberg-Osgood hardening. Adjust material parameters to see how the stress-strain relationship changes.">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Young&apos;s Modulus E: {youngsMod} GPa
          </SimulationLabel>
          <Slider
            min={50}
            max={400}
            step={10}
            value={[youngsMod]}
            onValueChange={([v]) => setYoungsMod(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>
            Yield Stress &sigma;<sub>Y</sub>: {yieldStress.toFixed(2)} GPa
          </SimulationLabel>
          <Slider
            min={0.05}
            max={1.0}
            step={0.01}
            value={[yieldStress]}
            onValueChange={([v]) => setYieldStress(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>
            Hardening Exponent n: {hardeningExp}
          </SimulationLabel>
          <Slider
            min={2}
            max={20}
            step={1}
            value={[hardeningExp]}
            onValueChange={([v]) => setHardeningExp(v)}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>
            Max Strain: {(maxStrain * 100).toFixed(1)}%
          </SimulationLabel>
          <Slider
            min={0.01}
            max={0.15}
            step={0.005}
            value={[maxStrain]}
            onValueChange={([v]) => setMaxStrain(v)}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasChart
          data={plotData.data}
          layout={plotData.layout}
          style={{ width: '100%', height: 500 }}
        />
      </SimulationMain>

      <div className="mt-4 text-sm text-[var(--text-muted)] space-y-1">
        <p><strong className="text-[var(--text-muted)]">Blue line:</strong> Linear elastic response &sigma; = E&epsilon; (Hooke&apos;s law).</p>
        <p><strong className="text-[var(--text-muted)]">Red curve:</strong> Ramberg-Osgood model &epsilon; = &sigma;/E + 0.002(&sigma;/&sigma;<sub>Y</sub>)<sup>n</sup>. At low stress the two curves coincide; beyond yield the material hardens nonlinearly.</p>
        <p><strong className="text-[var(--text-muted)]">Diamond:</strong> Conventional yield point defined by the 0.2% offset method.</p>
      </div>
    </SimulationPanel>
  );
}
