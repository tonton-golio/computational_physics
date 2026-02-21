'use client';

import React, { useMemo, useState } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { linspace } from './ml-utils';

interface ActivationDef {
  name: string;
  fn: (x: number) => number;
  deriv: (x: number) => number;
  color: string;
}

const ACTIVATIONS: ActivationDef[] = [
  {
    name: 'ReLU',
    fn: (x) => Math.max(0, x),
    deriv: (x) => (x > 0 ? 1 : 0),
    color: '#3b82f6',
  },
  {
    name: 'Sigmoid',
    fn: (x) => 1 / (1 + Math.exp(-x)),
    deriv: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
    color: '#ef4444',
  },
  {
    name: 'Tanh',
    fn: (x) => Math.tanh(x),
    deriv: (x) => 1 - Math.tanh(x) ** 2,
    color: '#10b981',
  },
  {
    name: 'Leaky ReLU',
    fn: (x) => (x > 0 ? x : 0.1 * x),
    deriv: (x) => (x > 0 ? 1 : 0.1),
    color: '#f59e0b',
  },
  {
    name: 'Swish',
    fn: (x) => x / (1 + Math.exp(-x)),
    deriv: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s + x * s * (1 - s);
    },
    color: '#8b5cf6',
  },
];

export default function ActivationGallery(): React.ReactElement {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(ACTIVATIONS.map((a) => a.name)),
  );

  const xVals = useMemo(() => linspace(-5, 5, 300), []);

  const fnTraces = useMemo(
    () =>
      ACTIVATIONS.filter((a) => selected.has(a.name)).map((a) => ({
        x: xVals,
        y: xVals.map(a.fn),
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: a.color, width: 2 },
        name: a.name,
      })),
    [xVals, selected],
  );

  const derivTraces = useMemo(
    () =>
      ACTIVATIONS.filter((a) => selected.has(a.name)).map((a) => ({
        x: xVals,
        y: xVals.map(a.deriv),
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: a.color, width: 2 },
        name: `${a.name}'`,
      })),
    [xVals, selected],
  );

  const toggleAct = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        if (next.size > 1) next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Activation Function Gallery
      </h3>

      <div className="mb-4 flex flex-wrap gap-2">
        {ACTIVATIONS.map((a) => (
          <button
            key={a.name}
            onClick={() => toggleAct(a.name)}
            className="rounded px-3 py-1.5 text-xs font-medium transition-opacity"
            style={{
              backgroundColor: a.color,
              opacity: selected.has(a.name) ? 1 : 0.3,
              color: 'white',
            }}
          >
            {a.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">
            Activation functions
          </p>
          <CanvasChart
            data={fnTraces}
            layout={{
              xaxis: { title: { text: 'x' } },
              yaxis: { title: { text: 'f(x)' } },
              margin: { t: 20, r: 20, b: 45, l: 55 },
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-sm text-[var(--text-muted)]">
            Derivatives
          </p>
          <CanvasChart
            data={derivTraces}
            layout={{
              xaxis: { title: { text: 'x' } },
              yaxis: { title: { text: "f'(x)" } },
              margin: { t: 20, r: 20, b: 45, l: 55 },
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
      </div>

      <div className="mt-3 text-xs text-[var(--text-muted)]">
        ReLU has constant gradient of 1 for positive inputs but kills negative ones (dead neurons).
        Sigmoid/Tanh have vanishing gradients for large |x|.
        Leaky ReLU and Swish maintain gradient flow throughout.
      </div>
    </div>
  );
}
