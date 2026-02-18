'use client';

import { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { Slider } from '@/components/ui/slider';
import { usePlotlyTheme } from '@/lib/plotly-theme';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function PhaseSpaceStates() {
  const [radius, setRadius] = useState(2.2);
  const [phase, setPhase] = useState(0.7);
  const { mergeLayout } = usePlotlyTheme();

  const { circleX, circleY, coherent, numberRing } = useMemo(() => {
    const t = Array.from({ length: 200 }, (_, i) => (2 * Math.PI * i) / 199);
    const circleX = t.map((v) => Math.cos(v));
    const circleY = t.map((v) => Math.sin(v));

    const cx = radius * Math.cos(phase);
    const cy = radius * Math.sin(phase);
    const coherent = { x: cx, y: cy };

    const nr = Math.sqrt(Math.max(radius, 0.2));
    const numberRing = {
      x: t.map((v) => nr * Math.cos(v)),
      y: t.map((v) => nr * Math.sin(v)),
    };

    return { circleX, circleY, coherent, numberRing };
  }, [radius, phase]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8 space-y-4">
      <h3 className="text-xl font-semibold text-[var(--text-strong)]">Phase-Space States: Coherent vs Number</h3>
      <p className="text-sm text-[var(--text-muted)]">
        Coherent states are localized displacements in phase space, while number states distribute isotropically by phase.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Coherent amplitude |alpha|: {radius.toFixed(2)}</label>
          <Slider min={0.2} max={4.5} step={0.05} value={[radius]} onValueChange={([v]) => setRadius(v)} />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Phase arg(alpha): {phase.toFixed(2)} rad</label>
          <Slider min={0} max={6.28} step={0.02} value={[phase]} onValueChange={([v]) => setPhase(v)} />
        </div>
      </div>
      <Plot
        data={[
          { x: circleX, y: circleY, type: 'scatter', mode: 'lines', line: { color: '#475569', width: 1 }, name: 'Vacuum reference' },
          { x: numberRing.x, y: numberRing.y, type: 'scatter', mode: 'lines', line: { color: '#10b981', width: 2, dash: 'dot' }, name: 'Number-state phase ring' },
          { x: [coherent.x], y: [coherent.y], type: 'scatter', mode: 'markers', marker: { size: 11, color: '#3b82f6' }, name: 'Coherent state centroid' },
        ]}
        layout={mergeLayout({
          title: { text: 'State geometry in quadrature space (q, p)' },
          margin: { t: 40, r: 20, b: 40, l: 50 },
          xaxis: { title: { text: 'q' }, range: [-5, 5] },
          yaxis: { title: { text: 'p' }, range: [-5, 5], scaleanchor: 'x' },
          height: 430,
        })}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: 430 }}
      />
    </div>
  );
}
