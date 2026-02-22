"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Cross-sectional view of a glacier showing the ice flow velocity profile and
 * shear stress distribution. Uses Glen's flow law (power-law n~3) to compute
 * the velocity profile driven by gravity on a slope.
 *
 * For a parallel slab of thickness H on slope angle theta:
 *   tau(y) = rho * g * sin(theta) * (H - y)
 *   du/dy  = A * tau^n
 *   u(y)   = A*(rho*g*sin(theta))^n / (n+1) * (H^(n+1) - (H-y)^(n+1))
 */

const rhoIce = 917; // kg/m^3
const g0 = 9.81;
const A = 2.4e-24; // Pa^-3 s^-1 (Glen's law rate factor at -10C)

export default function GlacierCrossSection({}: SimulationComponentProps) {
  const [slopeDeg, setSlopeDeg] = useState(5);
  const [thickness, setThickness] = useState(200); // m
  const [nGlen, setNGlen] = useState(3);

  const { velTraces, stressTraces } = useMemo(() => {
    const N = 200;
    const theta = (slopeDeg * Math.PI) / 180;
    const H = thickness;
    const tau0 = rhoIce * g0 * Math.sin(theta); // Pa/m
    const n = nGlen;

    const yArr: number[] = [];
    const velArr: number[] = [];
    const tauArr: number[] = [];
    const velNewt: number[] = [];

    for (let i = 0; i <= N; i++) {
      const y = (i / N) * H; // height above bed
      yArr.push(y);

      // Shear stress
      const tau = tau0 * (H - y);
      tauArr.push(tau / 1000); // kPa

      // Velocity (Glen's law)
      const uGlen =
        (A * Math.pow(tau0, n) / (n + 1)) *
        (Math.pow(H, n + 1) - Math.pow(H - y, n + 1));
      velArr.push(uGlen * 3.156e7); // m/yr

      // Newtonian (n=1) for comparison
      const uNewt =
        (A * tau0 / 2) * (2 * H * y - y * y);
      velNewt.push(uNewt * 3.156e7);
    }

    const velTraces = [
      {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: velArr,
        y: yArr,
        name: `Glen's law (n=${n})`,
        line: { color: '#3b82f6', width: 2.5 },
      },
      {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: velNewt,
        y: yArr,
        name: 'Newtonian (n=1)',
        line: { color: '#8fa3c4', width: 2, dash: 'dash' as const },
      },
    ];

    const stressTraces = [
      {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: tauArr,
        y: yArr,
        name: 'Shear stress',
        line: { color: '#ef4444', width: 2 },
      },
    ];

    return { velTraces, stressTraces };
  }, [slopeDeg, thickness, nGlen]);

  return (
    <SimulationPanel title="Glacier Cross-Section: Velocity and Shear Stress" caption="A glacier of uniform thickness on a slope. Glen's flow law (n\u22483) produces a \u201cplug-like\u201d profile: most shearing happens near the bed, while the upper ice moves as a nearly rigid block. Compare with the parabolic Newtonian profile.">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Slope angle: {slopeDeg.toFixed(1)}&deg;
          </SimulationLabel>
          <Slider min={1} max={15} step={0.5} value={[slopeDeg]}
            onValueChange={([v]) => setSlopeDeg(v)} className="w-full" />
        </div>
        <div>
          <SimulationLabel>
            Ice thickness: {thickness} m
          </SimulationLabel>
          <Slider min={50} max={500} step={10} value={[thickness]}
            onValueChange={([v]) => setThickness(v)} className="w-full" />
        </div>
        <div>
          <SimulationLabel>
            Glen exponent n: {nGlen}
          </SimulationLabel>
          <Slider min={1} max={5} step={1} value={[nGlen]}
            onValueChange={([v]) => setNGlen(v)} className="w-full" />
        </div>
      </SimulationConfig>
      <SimulationMain>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <CanvasChart
            data={velTraces}
            layout={{
              xaxis: { title: { text: 'Velocity (m/yr)' } },
              yaxis: { title: { text: 'Height above bed (m)' } },
            }}
            style={{ width: '100%', height: 380 }}
          />
          <CanvasChart
            data={stressTraces}
            layout={{
              xaxis: { title: { text: 'Shear stress (kPa)' } },
              yaxis: { title: { text: 'Height above bed (m)' } },
            }}
            style={{ width: '100%', height: 380 }}
          />
        </div>
      </SimulationMain>
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Left: velocity profile (blue = Glen, dashed = Newtonian). The non-linear
        rheology concentrates shear near the bed. Right: shear stress increases
        linearly from zero at the surface to &tau;<sub>b</sub> = &rho;gH sin&theta; at the bed.
      </p>
    </SimulationPanel>
  );
}
