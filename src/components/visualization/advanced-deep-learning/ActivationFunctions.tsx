'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ActivationFunctionsProps {
  id?: string;
}

type ActivationName = 'relu' | 'sigmoid' | 'tanh' | 'leaky-relu' | 'softmax' | 'swish';

const ACTIVATION_INFO: Record<ActivationName, { label: string; color: string; description: string }> = {
  relu: {
    label: 'ReLU',
    color: '#ff6b6b',
    description: 'f(x) = max(0, x). Most widely used. Fast to compute. Can suffer from dying neurons.',
  },
  sigmoid: {
    label: 'Sigmoid',
    color: '#4ecdc4',
    description: 'f(x) = 1 / (1 + e^{-x}). Output in (0,1). Used for binary classification output layers.',
  },
  tanh: {
    label: 'Tanh',
    color: '#45b7d1',
    description: 'f(x) = tanh(x). Output in (-1,1). Zero-centered, often preferred over sigmoid in hidden layers.',
  },
  'leaky-relu': {
    label: 'Leaky ReLU',
    color: '#f9ca24',
    description: 'f(x) = x if x > 0, else 0.01x. Fixes the dying ReLU problem with a small negative slope.',
  },
  softmax: {
    label: 'Softmax (vs 0)',
    color: '#6c5ce7',
    description: 'f(x_i) = e^{x_i} / sum(e^{x_j}). Outputs probabilities. Here shown as softmax of [x, 0].',
  },
  swish: {
    label: 'Swish',
    color: '#e17055',
    description: 'f(x) = x * sigmoid(x). Smooth, non-monotonic. Used in EfficientNet and other modern architectures.',
  },
};

function computeActivation(name: ActivationName, x: number): number {
  switch (name) {
    case 'relu':
      return Math.max(0, x);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-x));
    case 'tanh':
      return Math.tanh(x);
    case 'leaky-relu':
      return x > 0 ? x : 0.01 * x;
    case 'softmax':
      // softmax of [x, 0] -> e^x / (e^x + 1)
      return Math.exp(x) / (Math.exp(x) + 1);
    case 'swish':
      return x / (1 + Math.exp(-x));
    default:
      return x;
  }
}

function computeDerivative(name: ActivationName, x: number): number {
  switch (name) {
    case 'relu':
      return x > 0 ? 1 : 0;
    case 'sigmoid': {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    }
    case 'tanh': {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
    case 'leaky-relu':
      return x > 0 ? 1 : 0.01;
    case 'softmax': {
      const s = Math.exp(x) / (Math.exp(x) + 1);
      return s * (1 - s);
    }
    case 'swish': {
      const sig = 1 / (1 + Math.exp(-x));
      return sig + x * sig * (1 - sig);
    }
    default:
      return 1;
  }
}

const ALL_ACTIVATION_NAMES: ActivationName[] = ['relu', 'sigmoid', 'tanh', 'leaky-relu', 'softmax', 'swish'];

export default function ActivationFunctions({ id }: ActivationFunctionsProps) {
  const [selected, setSelected] = useState<Set<ActivationName>>(
    new Set(['relu', 'sigmoid', 'tanh'])
  );
  const [showDerivatives, setShowDerivatives] = useState(false);
  const [xRange, setXRange] = useState(5);

  const xValues = useMemo(() => {
    const xs: number[] = [];
    const steps = 200;
    for (let i = 0; i <= steps; i++) {
      xs.push(-xRange + (2 * xRange * i) / steps);
    }
    return xs;
  }, [xRange]);

  const plotData = useMemo(() => {
    const traces: any[] = [];
    for (const name of ALL_ACTIVATION_NAMES) {
      if (!selected.has(name)) continue;
      const info = ACTIVATION_INFO[name];
      traces.push({
        type: 'scatter' as const,
        x: xValues,
        y: xValues.map((x) => computeActivation(name, x)),
        mode: 'lines' as const,
        line: { color: info.color, width: 2.5 },
        name: info.label,
      });
      if (showDerivatives) {
        traces.push({
          type: 'scatter' as const,
          x: xValues,
          y: xValues.map((x) => computeDerivative(name, x)),
          mode: 'lines' as const,
          line: { color: info.color, width: 1.5, dash: 'dash' },
          name: `${info.label}'`,
        });
      }
    }
    return traces;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, showDerivatives, xValues]);

  const toggleActivation = (name: ActivationName) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">
        Activation Functions Comparison
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-3">
          <p className="text-sm text-gray-400 font-semibold">Select activations:</p>
          {ALL_ACTIVATION_NAMES.map((name) => {
            const info = ACTIVATION_INFO[name];
            return (
              <label
                key={name}
                className="flex items-center gap-2 text-sm cursor-pointer"
                style={{ color: selected.has(name) ? info.color : '#666' }}
              >
                <input
                  type="checkbox"
                  checked={selected.has(name)}
                  onChange={() => toggleActivation(name)}
                  className="accent-blue-500"
                />
                {info.label}
              </label>
            );
          })}

          <hr className="border-gray-700 my-3" />

          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showDerivatives}
              onChange={(e) => setShowDerivatives(e.target.checked)}
              className="accent-blue-500"
            />
            Show Derivatives (dashed)
          </label>

          <div>
            <label className="block text-sm text-gray-400 mb-1">
              X Range: [-{xRange}, {xRange}]
            </label>
            <input
              type="range"
              min={2}
              max={10}
              step={1}
              value={xRange}
              onChange={(e) => setXRange(parseInt(e.target.value))}
              className="w-full accent-blue-500"
            />
          </div>

          {/* Info box */}
          <div className="mt-3 p-3 bg-[#0a0a15] rounded text-xs text-gray-400 space-y-2">
            {ALL_ACTIVATION_NAMES.filter((n) => selected.has(n)).map((name) => (
              <div key={name}>
                <span className="font-semibold" style={{ color: ACTIVATION_INFO[name].color }}>
                  {ACTIVATION_INFO[name].label}:
                </span>{' '}
                {ACTIVATION_INFO[name].description}
              </div>
            ))}
          </div>
        </div>

        {/* Plot */}
        <div className="lg:col-span-2">
          <Plot
            data={plotData}
            layout={{
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(15,15,25,1)',
              font: { color: '#ccc' },
              xaxis: {
                title: { text: 'x' },
                gridcolor: 'rgba(255,255,255,0.1)',
                zerolinecolor: 'rgba(255,255,255,0.3)',
                zerolinewidth: 1,
              },
              yaxis: {
                title: { text: 'f(x)' },
                gridcolor: 'rgba(255,255,255,0.1)',
                zerolinecolor: 'rgba(255,255,255,0.3)',
                zerolinewidth: 1,
              },
              legend: {
                bgcolor: 'rgba(0,0,0,0.5)',
                font: { color: '#ccc', size: 11 },
              },
              margin: { t: 30, b: 50, l: 60, r: 30 },
              autosize: true,
            }}
            useResizeHandler
            style={{ width: '100%', height: '450px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>
    </div>
  );
}
