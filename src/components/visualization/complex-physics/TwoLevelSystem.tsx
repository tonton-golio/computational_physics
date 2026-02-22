"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationConfig, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

function computeTwoLevel(epsilon: number, tRange: number[]) {
  const Z: number[] = [];
  const E: number[] = [];
  const Cv: number[] = [];
  const S: number[] = [];

  for (const T of tRange) {
    const beta = 1 / T;
    const expFactor = Math.exp(-beta * epsilon);
    const z = 1 + expFactor;
    const avgE = epsilon * expFactor / z;
    const avgE2 = epsilon * epsilon * expFactor / z;
    const cv = (avgE2 - avgE * avgE) / (T * T);
    const s = Math.log(z) + avgE / T;

    Z.push(z);
    E.push(avgE);
    Cv.push(cv);
    S.push(s);
  }

  return { Z, E, Cv, S };
}

export default function TwoLevelSystem({}: SimulationComponentProps) {
  const [epsilon, setEpsilon] = useState(1.0);

  const tRange = useMemo(() => {
    const pts: number[] = [];
    for (let i = 1; i <= 200; i++) {
      pts.push(0.05 * i);
    }
    return pts;
  }, []);

  const data = useMemo(() => computeTwoLevel(epsilon, tRange), [epsilon, tRange]);

  const lineStyle = { color: '#8b5cf6', width: 2 };

  return (
    <SimulationPanel title="Two-Level System Thermodynamics">
      <SimulationConfig>
        <div>
          <SimulationLabel>
            Energy gap {'\u03B5'}: {epsilon.toFixed(2)}
          </SimulationLabel>
          <Slider
            min={0.1}
            max={5}
            step={0.1}
            value={[epsilon]}
            onValueChange={([v]) => setEpsilon(v)}
            className="w-full"
          />
        </div>
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CanvasChart
          data={[{
            x: tRange,
            y: data.Z,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: lineStyle,
            name: 'Z(T)',
          }]}
          layout={{
            title: { text: 'Partition Function Z(T)', font: { size: 13 } },
            xaxis: { title: { text: 'T' } },
            yaxis: { title: { text: 'Z' } },
            showlegend: false,
            margin: { t: 35, r: 15, b: 45, l: 50 },
          }}
          style={{ width: '100%', height: 250 }}
        />
        <CanvasChart
          data={[{
            x: tRange,
            y: data.E,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#3b82f6', width: 2 },
            name: '\u27E8E\u27E9(T)',
          }]}
          layout={{
            title: { text: 'Average Energy \u27E8E\u27E9(T)', font: { size: 13 } },
            xaxis: { title: { text: 'T' } },
            yaxis: { title: { text: '\u27E8E\u27E9' } },
            showlegend: false,
            margin: { t: 35, r: 15, b: 45, l: 50 },
          }}
          style={{ width: '100%', height: 250 }}
        />
        <CanvasChart
          data={[{
            x: tRange,
            y: data.Cv,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#ef4444', width: 2 },
            name: 'C_V(T)',
          }]}
          layout={{
            title: { text: 'Heat Capacity C\u1D65(T)', font: { size: 13 } },
            xaxis: { title: { text: 'T' } },
            yaxis: { title: { text: 'C\u1D65' } },
            showlegend: false,
            margin: { t: 35, r: 15, b: 45, l: 50 },
          }}
          style={{ width: '100%', height: 250 }}
        />
        <CanvasChart
          data={[{
            x: tRange,
            y: data.S,
            type: 'scatter' as const,
            mode: 'lines' as const,
            line: { color: '#10b981', width: 2 },
            name: 'S(T)',
          }]}
          layout={{
            title: { text: 'Entropy S(T)', font: { size: 13 } },
            xaxis: { title: { text: 'T' } },
            yaxis: { title: { text: 'S' } },
            showlegend: false,
            margin: { t: 35, r: 15, b: 45, l: 50 },
          }}
          style={{ width: '100%', height: 250 }}
        />
        </div>
      </SimulationMain>
    </SimulationPanel>
  );
}
