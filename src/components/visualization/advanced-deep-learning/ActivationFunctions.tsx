'use client';

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';

interface ActivationFunctionsProps {
  id?: string;
}

type ActivationName = 'relu' | 'sigmoid' | 'tanh' | 'leaky-relu' | 'softmax' | 'swish';

const ACTIVATION_INFO: Record<ActivationName, {
  label: string;
  color: string;
  description: string;
  gradientNote: string;
  whyItMatters: string;
}> = {
  relu: {
    label: 'ReLU',
    color: '#ff6b6b',
    description: 'f(x) = max(0, x). Most widely used. Fast to compute. Can suffer from dying neurons.',
    gradientNote: 'Gradient is exactly 1 for x > 0 and 0 for x < 0. No vanishing gradient in the positive region, but "dead neurons" produce zero gradient forever.',
    whyItMatters: 'Default choice for hidden layers in most architectures. Simple, fast, and effective.',
  },
  sigmoid: {
    label: 'Sigmoid',
    color: '#4ecdc4',
    description: 'f(x) = 1 / (1 + e^{-x}). Output in (0,1). Used for binary classification output layers.',
    gradientNote: 'Maximum gradient of 0.25 at x=0. Saturates for |x| > 4, causing vanishing gradients in deep networks.',
    whyItMatters: 'Used in output layers for binary classification and in gates (LSTM, GRU). Avoid in hidden layers.',
  },
  tanh: {
    label: 'Tanh',
    color: '#45b7d1',
    description: 'f(x) = tanh(x). Output in (-1,1). Zero-centered, often preferred over sigmoid in hidden layers.',
    gradientNote: 'Maximum gradient of 1.0 at x=0. Better gradient flow than sigmoid but still saturates for large |x|.',
    whyItMatters: 'Zero-centered outputs help with gradient flow. Used in RNN hidden states and some normalization layers.',
  },
  'leaky-relu': {
    label: 'Leaky ReLU',
    color: '#f9ca24',
    description: 'f(x) = x if x > 0, else 0.01x. Fixes the dying ReLU problem with a small negative slope.',
    gradientNote: 'Gradient is 1 for x > 0 and 0.01 for x < 0. Never fully zero, preventing dead neurons.',
    whyItMatters: 'Drop-in replacement for ReLU when dead neurons are a problem. Parametric version (PReLU) learns the slope.',
  },
  softmax: {
    label: 'Softmax (vs 0)',
    color: '#6c5ce7',
    description: 'f(x_i) = e^{x_i} / sum(e^{x_j}). Outputs probabilities. Here shown as softmax of [x, 0].',
    gradientNote: 'Gradient depends on all inputs jointly. Saturates when one logit dominates, similar to sigmoid.',
    whyItMatters: 'Standard output activation for multi-class classification. Converts logits to probabilities summing to 1.',
  },
  swish: {
    label: 'Swish',
    color: '#e17055',
    description: 'f(x) = x * sigmoid(x). Smooth, non-monotonic. Used in EfficientNet and other modern architectures.',
    gradientNote: 'Smooth gradient everywhere. Slight negative region allows "forgetting" of some signals. Non-monotonic near x = -1.',
    whyItMatters: 'Often outperforms ReLU in deep networks. Smooth gradient landscape helps optimization.',
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

export default function ActivationFunctions({ id: _id }: ActivationFunctionsProps) {
  const [selected, setSelected] = useState<Set<ActivationName>>(
    new Set(['relu', 'sigmoid', 'tanh'])
  );
  const [showDerivatives, setShowDerivatives] = useState(false);
  const [showGradientRegions, setShowGradientRegions] = useState(false);
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

    // Add vanishing gradient highlight regions
    if (showGradientRegions) {
      for (const name of ALL_ACTIVATION_NAMES) {
        if (!selected.has(name)) continue;
        if (name === 'sigmoid' || name === 'tanh' || name === 'softmax') {
          // Highlight saturation regions where gradient < 0.05
          const info = ACTIVATION_INFO[name];
          traces.push({
            type: 'scatter' as const,
            x: xValues.filter(x => Math.abs(computeDerivative(name, x)) < 0.05),
            y: xValues.filter(x => Math.abs(computeDerivative(name, x)) < 0.05).map(x => computeActivation(name, x)),
            mode: 'markers' as const,
            marker: { color: info.color, size: 3, opacity: 0.3, symbol: 'square' },
            name: `${info.label} vanishing region`,
            showlegend: false,
          });
        }
        if (name === 'relu') {
          // Highlight dead region (x < 0)
          traces.push({
            type: 'scatter' as const,
            x: xValues.filter(x => x < 0),
            y: xValues.filter(x => x < 0).map(() => 0),
            mode: 'markers' as const,
            marker: { color: '#ff6b6b', size: 3, opacity: 0.3, symbol: 'square' },
            name: 'ReLU dead region',
            showlegend: false,
          });
        }
      }
    }

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
  }, [selected, showDerivatives, showGradientRegions, xValues]);

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
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Activation Functions Comparison
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-3">
          <p className="text-sm text-[var(--text-muted)] font-semibold">Select activations:</p>
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

          <hr className="border-[var(--border-strong)] my-3" />

          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer">
            <input
              type="checkbox"
              checked={showDerivatives}
              onChange={(e) => setShowDerivatives(e.target.checked)}
              className="accent-blue-500"
            />
            Show Derivatives (dashed)
          </label>

          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer">
            <input
              type="checkbox"
              checked={showGradientRegions}
              onChange={(e) => setShowGradientRegions(e.target.checked)}
              className="accent-blue-500"
            />
            Highlight Vanishing Gradient Regions
          </label>

          <div>
            <label className="block text-sm text-[var(--text-muted)] mb-1">
              X Range: [-{xRange}, {xRange}]
            </label>
            <Slider
              min={2}
              max={10}
              step={1}
              value={[xRange]}
              onValueChange={([v]) => setXRange(v)}
              className="w-full"
            />
          </div>

          {/* Gradient annotation panel */}
          <div className="mt-3 p-3 bg-[var(--surface-2)] rounded text-xs text-[var(--text-muted)] space-y-2">
            {ALL_ACTIVATION_NAMES.filter((n) => selected.has(n)).map((name) => (
              <div key={name} className="space-y-1">
                <span className="font-semibold" style={{ color: ACTIVATION_INFO[name].color }}>
                  {ACTIVATION_INFO[name].label}:
                </span>{' '}
                {ACTIVATION_INFO[name].description}
                {showDerivatives && (
                  <div className="ml-2 mt-1 text-[var(--text-muted)] italic">
                    Gradient: {ACTIVATION_INFO[name].gradientNote}
                  </div>
                )}
                <div className="ml-2 text-blue-400/70">
                  {ACTIVATION_INFO[name].whyItMatters}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Plot */}
        <div className="lg:col-span-2">
          <CanvasChart
            data={plotData}
            layout={{
              xaxis: {
                title: { text: 'x' },
                zerolinewidth: 1,
              },
              yaxis: {
                title: { text: showDerivatives ? 'f(x) / f\'(x)' : 'f(x)' },
                zerolinewidth: 1,
              },
              margin: { t: 30, b: 50, l: 60, r: 30 },
              autosize: true,
            }}
            style={{ width: '100%', height: '450px' }}
          />
        </div>
      </div>
    </div>
  );
}
