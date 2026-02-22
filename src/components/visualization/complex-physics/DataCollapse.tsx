"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

// Pre-computed susceptibility data for 2D Ising model near Tc
// Using known scaling: chi ~ L^{gamma/nu} * f(t * L^{1/nu})
// with gamma/nu = 7/4 = 1.75 and 1/nu = 1.0 for 2D Ising
const Tc = 2.269;
const TRUE_GAMMA_NU = 1.75;
const TRUE_INV_NU = 1.0;

function generateSusceptibilityData(L: number) {
  const tVals: number[] = [];
  const chiVals: number[] = [];
  const nPts = 100;

  for (let i = 0; i <= nPts; i++) {
    const T = 1.5 + (2.0 * i) / nPts; // T from 1.5 to 3.5
    const t = (T - Tc) / Tc;
    const x = t * Math.pow(L, TRUE_INV_NU);

    // Universal scaling function: Gaussian peak shape
    const scalingFn = Math.exp(-0.5 * x * x) * (1 + 0.1 * x);
    const chi = Math.pow(L, TRUE_GAMMA_NU) * Math.abs(scalingFn);

    tVals.push(T);
    chiVals.push(chi);
  }

  return { T: tVals, chi: chiVals };
}

const SIZES = [16, 32, 64, 128];
const COLORS = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b'];

export default function DataCollapse({}: SimulationComponentProps) {
  const [gammaOverNu, setGammaOverNu] = useState(1.75);
  const [invNu, setInvNu] = useState(1.0);

  const rawData = useMemo(
    () => SIZES.map((L) => ({ L, ...generateSusceptibilityData(L) })),
    []
  );

  const collapsedData = useMemo(() => {
    return rawData.map((d) => {
      const xCollapse: number[] = [];
      const yCollapse: number[] = [];
      for (let i = 0; i < d.T.length; i++) {
        const t = (d.T[i] - Tc) / Tc;
        const x = t * Math.pow(d.L, invNu);
        const y = d.chi[i] / Math.pow(d.L, gammaOverNu);
        xCollapse.push(x);
        yCollapse.push(y);
      }
      return { L: d.L, x: xCollapse, y: yCollapse };
    });
  }, [rawData, gammaOverNu, invNu]);

  return (
    <SimulationPanel title="Data Collapse" caption="Adjust the exponents until all four curves collapse onto a single master curve. The collapse is perfect at the exact 2D Ising values.">
      <SimulationConfig>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <SimulationLabel>
              {'\u03B3'}/{'\u03BD'}: {gammaOverNu.toFixed(2)} (exact: 1.75)
            </SimulationLabel>
            <Slider
              min={0.5}
              max={4}
              step={0.05}
              value={[gammaOverNu]}
              onValueChange={([v]) => setGammaOverNu(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>
              1/{'\u03BD'}: {invNu.toFixed(2)} (exact: 1.00)
            </SimulationLabel>
            <Slider
              min={0.5}
              max={2}
              step={0.05}
              value={[invNu]}
              onValueChange={([v]) => setInvNu(v)}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <CanvasChart
          data={rawData.map((d, idx) => ({
            x: d.T,
            y: d.chi,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: COLORS[idx], width: 1.5 },
            name: `L = ${d.L}`,
          }))}
          layout={{
            title: { text: 'Raw \u03C7(T)', font: { size: 13 } },
            xaxis: { title: { text: 'Temperature T' } },
            yaxis: { title: { text: '\u03C7' } },
            showlegend: true,
            margin: { t: 35, r: 15, b: 45, l: 55 },
          }}
          style={{ width: '100%', height: 350 }}
        />
        <CanvasChart
          data={collapsedData.map((d, idx) => ({
            x: d.x,
            y: d.y,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: COLORS[idx], width: 1.5 },
            name: `L = ${d.L}`,
          }))}
          layout={{
            title: { text: 'Data Collapse: \u03C7/L^{\u03B3/\u03BD} vs tL^{1/\u03BD}', font: { size: 13 } },
            xaxis: { title: { text: 't \u00B7 L^{1/\u03BD}' } },
            yaxis: { title: { text: '\u03C7 / L^{\u03B3/\u03BD}' } },
            showlegend: true,
            margin: { t: 35, r: 15, b: 45, l: 55 },
          }}
          style={{ width: '100%', height: 350 }}
        />
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
