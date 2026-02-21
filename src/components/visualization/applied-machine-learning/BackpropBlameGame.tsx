'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Slider } from '@/components/ui/slider';

type Activation = 'relu' | 'sigmoid';

interface LayerConfig {
  size: number;
  activation: Activation;
  frozen: boolean;
}

const LAYER_DEFAULTS: LayerConfig[] = [
  { size: 2, activation: 'relu', frozen: false },
  { size: 4, activation: 'relu', frozen: false },
  { size: 4, activation: 'relu', frozen: false },
  { size: 1, activation: 'relu', frozen: false },
];

function _activationDerivative(act: Activation, preact: number): number {
  switch (act) {
    case 'relu':
      return preact > 0 ? 1 : 0;
    case 'sigmoid': {
      const s = 1 / (1 + Math.exp(-preact));
      return s * (1 - s); // max ~0.25
    }
  }
}

/**
 * Compute gradient magnitudes for each layer flowing backward from the output.
 * Simplified model: each layer multiplies the gradient by the average derivative.
 * This captures vanishing/exploding gradient dynamics without full weight matrices.
 */
function computeGradients(
  layers: LayerConfig[],
  outputLoss: number,
): number[] {
  const gradients: number[] = new Array(layers.length).fill(0);
  gradients[layers.length - 1] = outputLoss;

  for (let i = layers.length - 2; i >= 0; i--) {
    if (layers[i + 1].frozen) {
      gradients[i] = 0;
      continue;
    }
    // Average derivative across neurons, assuming typical pre-activations
    const act = layers[i + 1].activation;
    const avgDeriv =
      act === 'relu'
        ? 0.5 // ~50% of neurons active
        : 0.2; // sigmoid derivative is ~0.25 max, often smaller
    gradients[i] = gradients[i + 1] * avgDeriv * Math.sqrt(layers[i + 1].size / 4);
  }

  return gradients;
}

export default function BackpropBlameGame(): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loss, setLoss] = useState(1.0);
  const [layers, setLayers] = useState<LayerConfig[]>(LAYER_DEFAULTS.map((l) => ({ ...l })));
  const [animPhase, setAnimPhase] = useState(0); // 0-1 for backward flow animation
  const rafRef = useRef<number>(0);

  const gradients = useMemo(() => computeGradients(layers, loss), [layers, loss]);

  const toggleActivation = useCallback(
    (layerIdx: number) => {
      if (layerIdx === 0 || layerIdx === layers.length - 1) return; // don't toggle input/output
      setLayers((prev) =>
        prev.map((l, i) =>
          i === layerIdx
            ? { ...l, activation: l.activation === 'relu' ? 'sigmoid' : 'relu' }
            : l,
        ),
      );
    },
    [layers.length],
  );

  const toggleFreeze = useCallback((layerIdx: number) => {
    setLayers((prev) =>
      prev.map((l, i) => (i === layerIdx ? { ...l, frozen: !l.frozen } : l)),
    );
  }, []);

  // Animate backward flow
  useEffect(() => {
    let start: number | null = null;
    const duration = 2000;

    const tick = (ts: number) => {
      if (start === null) start = ts;
      const elapsed = ts - start;
      const phase = Math.min(1, elapsed / duration);
      setAnimPhase(phase);

      if (phase < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        // Restart after a pause
        setTimeout(() => {
          start = null;
          rafRef.current = requestAnimationFrame(tick);
        }, 1000);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  // Draw the network
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, W, H);

    const nLayers = layers.length;
    const layerSpacing = W / (nLayers + 1);
    const maxNeurons = Math.max(...layers.map((l) => l.size));
    const neuronRadius = Math.min(22, (H - 80) / (maxNeurons * 2 + 2));

    // Compute neuron positions
    const positions: { x: number; y: number }[][] = layers.map((layer, li) => {
      const x = layerSpacing * (li + 1);
      return Array.from({ length: layer.size }, (_, ni) => {
        const totalHeight = (layer.size - 1) * neuronRadius * 2.8;
        const y = H / 2 - totalHeight / 2 + ni * neuronRadius * 2.8;
        return { x, y };
      });
    });

    // Draw connections with gradient coloring
    for (let li = 0; li < nLayers - 1; li++) {
      const gradMag = gradients[li];
      const normalizedGrad = Math.min(1, Math.abs(gradMag) / (loss + 0.01));
      const backwardProgress = 1 - (li + 1) / nLayers;

      // Show arrows flowing backward during animation
      const showArrow = animPhase >= backwardProgress;
      const alpha = showArrow ? Math.min(1, normalizedGrad * 2 + 0.1) : 0.15;
      const width = showArrow ? Math.max(1, normalizedGrad * 6) : 1;

      for (const src of positions[li]) {
        for (const dst of positions[li + 1]) {
          // Connection line
          ctx.strokeStyle = showArrow
            ? `rgba(34, 211, 238, ${alpha})`
            : 'rgba(100, 116, 139, 0.2)';
          ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(src.x + neuronRadius, src.y);
          ctx.lineTo(dst.x - neuronRadius, dst.y);
          ctx.stroke();

          // Backward arrow indicator
          if (showArrow && normalizedGrad > 0.05) {
            const progress = Math.min(1, (animPhase - backwardProgress) * nLayers);
            const arrowX = dst.x - neuronRadius - (dst.x - src.x - 2 * neuronRadius) * progress;
            const arrowY = dst.y - (dst.y - src.y) * progress;

            ctx.fillStyle = `rgba(239, 68, 68, ${alpha})`;
            ctx.beginPath();
            ctx.arc(arrowX, arrowY, Math.max(2, width), 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }

    // Draw neurons
    for (let li = 0; li < nLayers; li++) {
      const layer = layers[li];
      const gradMag = gradients[li];
      const normalizedGrad = Math.min(1, Math.abs(gradMag) / (loss + 0.01));

      for (let ni = 0; ni < layer.size; ni++) {
        const { x, y } = positions[li][ni];

        // Glow effect based on gradient magnitude
        if (normalizedGrad > 0.1 && animPhase > (1 - (li + 1) / nLayers)) {
          const gradient = ctx.createRadialGradient(x, y, neuronRadius, x, y, neuronRadius * 2);
          gradient.addColorStop(0, `rgba(34, 211, 238, ${normalizedGrad * 0.3})`);
          gradient.addColorStop(1, 'rgba(34, 211, 238, 0)');
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(x, y, neuronRadius * 2, 0, Math.PI * 2);
          ctx.fill();
        }

        // Neuron circle
        const isOutput = li === nLayers - 1;
        const isInput = li === 0;
        ctx.fillStyle = layer.frozen
          ? '#6b7280'
          : isOutput
            ? '#ef4444'
            : isInput
              ? '#3b82f6'
              : layer.activation === 'sigmoid'
                ? '#f59e0b'
                : '#10b981';
        ctx.beginPath();
        ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
        ctx.fill();

        // Label
        ctx.fillStyle = 'white';
        ctx.font = `bold ${Math.max(10, neuronRadius * 0.6)}px system-ui`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const label = isInput ? 'x' : isOutput ? 'L' : layer.activation === 'relu' ? 'R' : 'S';
        ctx.fillText(label, x, y);
      }

      // Layer label
      ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'center';
      const layerLabels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
      ctx.fillText(layerLabels[li] || `Layer ${li}`, positions[li][0].x, H - 12);

      // Gradient magnitude
      if (li < nLayers - 1) {
        ctx.fillStyle = Math.abs(gradMag) < 0.05 ? '#ef4444' : 'rgba(148, 163, 184, 0.6)';
        ctx.font = '10px system-ui';
        ctx.fillText(`|g|=${Math.abs(gradMag).toFixed(3)}`, positions[li][0].x, 14);
      }
    }

    // Reset canvas dimensions (prevent accumulating DPR scaling)
    canvas.width = W;
    canvas.height = H;
  }, [layers, gradients, loss, animPhase]);

  const allSigmoid = layers.slice(1, -1).every((l) => l.activation === 'sigmoid');
  const vanishing = gradients[0] !== undefined && Math.abs(gradients[0]) < 0.01;

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6">
      <h3 className="mb-4 text-xl font-semibold text-[var(--text-strong)]">
        Backpropagation: The Blame Game
      </h3>

      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Output loss: {loss.toFixed(2)}</label>
          <Slider min={0.1} max={3.0} step={0.1} value={[loss]} onValueChange={([v]) => setLoss(v)} />
        </div>
        <div className="flex flex-wrap gap-2">
          {layers.slice(1, -1).map((l, i) => (
            <button
              key={i}
              onClick={() => toggleActivation(i + 1)}
              className={`rounded px-3 py-1 text-xs font-medium ${
                l.activation === 'relu'
                  ? 'bg-emerald-600/80 text-white'
                  : 'bg-amber-600/80 text-white'
              }`}
            >
              H{i + 1}: {l.activation.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          {layers.slice(1, -1).map((l, i) => (
            <button
              key={i}
              onClick={() => toggleFreeze(i + 1)}
              className={`rounded px-3 py-1 text-xs font-medium ${
                l.frozen
                  ? 'bg-gray-600 text-white'
                  : 'bg-[var(--surface-2,#27272a)] text-[var(--text-strong)]'
              }`}
            >
              {l.frozen ? 'Frozen' : 'Freeze'} H{i + 1}
            </button>
          ))}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={700}
        height={280}
        className="w-full rounded-xl bg-[var(--surface-2,#27272a)]"
        style={{ maxWidth: 700 }}
      />

      <div className="mt-2 text-xs text-[var(--text-muted)]">
        Click activation buttons to switch between ReLU (R) and Sigmoid (S).
        Arrow thickness = gradient magnitude. Red dots flow backward showing chain rule.
      </div>

      {vanishing && allSigmoid && (
        <div className="mt-3 rounded bg-red-900/30 p-3 text-sm text-red-300">
          Vanishing gradients detected. With all-sigmoid activations, each layer
          multiplies the gradient by at most 0.25 — after several layers, the
          gradient effectively disappears. Switch to ReLU to fix this.
        </div>
      )}

      {vanishing && !allSigmoid && layers.some((l) => l.frozen) && (
        <div className="mt-3 rounded bg-amber-900/30 p-3 text-sm text-amber-300">
          Frozen layers block gradient flow. Earlier layers cannot learn because
          the chain rule cannot pass through a frozen layer.
        </div>
      )}
    </div>
  );
}
