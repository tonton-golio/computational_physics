'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Slider } from '@/components/ui/slider';

function relu(x: number): number {
  return Math.max(0, x);
}
function reluDeriv(x: number): number {
  return x > 0 ? 1 : 0;
}

interface NetworkState {
  // Forward values
  input: number[];
  z1: number[]; // pre-activation hidden
  h1: number[]; // post-activation hidden
  z2: number; // pre-activation output
  output: number;
  loss: number;
  // Backward gradients
  dLoss_dOut: number;
  dOut_dZ2: number;
  dZ2_dH1: number[];
  dH1_dZ1: number[];
  dZ1_dInput: number[][];
}

const W1 = [
  [0.6, 0.3],
  [-0.4, 0.8],
  [0.5, -0.2],
];
const B1 = [0.1, -0.1, 0.2];
const W2 = [0.7, -0.5, 0.4];
const B2 = 0.1;
const TARGET = 1.0;

function computeNetwork(input: number[]): NetworkState {
  // Forward: 2 -> 3 -> 1
  const z1 = W1.map((w, j) => w[0] * input[0] + w[1] * input[1] + B1[j]);
  const h1 = z1.map(relu);
  const z2 = W2.reduce((a, w, j) => a + w * h1[j], 0) + B2;
  const output = z2; // linear output
  const loss = 0.5 * (output - TARGET) ** 2;

  // Backward
  const dLoss_dOut = output - TARGET;
  const dOut_dZ2 = 1; // linear
  const dZ2_dH1 = [...W2];
  const dH1_dZ1 = z1.map(reluDeriv);
  const dZ1_dInput = W1.map((w) => [...w]);

  return {
    input,
    z1,
    h1,
    z2,
    output,
    loss,
    dLoss_dOut,
    dOut_dZ2,
    dZ2_dH1,
    dH1_dZ1,
    dZ1_dInput,
  };
}

export default function ForwardBackwardPass(): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [x1, setX1] = useState(0.8);
  const [x2, setX2] = useState(0.3);
  const [phase, setPhase] = useState<'forward' | 'backward'>('forward');
  const [animProgress, setAnimProgress] = useState(1);
  const rafRef = useRef<number>(0);

  const net = useMemo(() => computeNetwork([x1, x2]), [x1, x2]);

  const animate = useCallback((direction: 'forward' | 'backward') => {
    setPhase(direction);
    setAnimProgress(0);
    let start: number | null = null;
    const duration = 1500;
    const tick = (ts: number) => {
      if (start === null) start = ts;
      const p = Math.min(1, (ts - start) / duration);
      setAnimProgress(p);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
    };
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  // Draw the network
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = container.clientWidth;
    const H = 320;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const layers = [2, 3, 1];
    const layerX = [W * 0.15, W * 0.45, W * 0.75];
    const nodeR = 24;

    // Compute positions
    const positions: { x: number; y: number }[][] = layers.map((size, li) => {
      return Array.from({ length: size }, (_, ni) => {
        const totalH = (size - 1) * 70;
        return { x: layerX[li], y: H / 2 - totalH / 2 + ni * 70 };
      });
    });

    // Values to display
    const nodeValues: string[][] = [
      net.input.map((v) => v.toFixed(2)),
      net.h1.map((v) => v.toFixed(2)),
      [net.output.toFixed(2)],
    ];

    // Gradient magnitudes for backward
    const layerGradMag = [
      // Input layer: chain rule all the way back
      net.input.map((_, i) => {
        let g = 0;
        for (let j = 0; j < 3; j++) {
          g += Math.abs(
            net.dLoss_dOut *
              net.dOut_dZ2 *
              net.dZ2_dH1[j] *
              net.dH1_dZ1[j] *
              net.dZ1_dInput[j][i],
          );
        }
        return g;
      }),
      // Hidden layer
      net.h1.map((_, j) =>
        Math.abs(net.dLoss_dOut * net.dOut_dZ2 * net.dZ2_dH1[j] * net.dH1_dZ1[j]),
      ),
      [Math.abs(net.dLoss_dOut)],
    ];

    // Draw connections
    for (let li = 0; li < layers.length - 1; li++) {
      for (let si = 0; si < layers[li]; si++) {
        for (let di = 0; di < layers[li + 1]; di++) {
          const src = positions[li][si];
          const dst = positions[li + 1][di];

          const layerProgress =
            phase === 'forward'
              ? (animProgress - li * 0.4) / 0.6
              : (animProgress - (1 - li) * 0.4) / 0.6;
          const showConn = layerProgress > 0;

          ctx.strokeStyle = showConn
            ? phase === 'forward'
              ? 'rgba(59, 130, 246, 0.6)'
              : 'rgba(239, 68, 68, 0.5)'
            : 'rgba(100, 116, 139, 0.2)';
          ctx.lineWidth = showConn ? 2 : 1;
          ctx.beginPath();
          ctx.moveTo(src.x + nodeR, src.y);
          ctx.lineTo(dst.x - nodeR, dst.y);
          ctx.stroke();

          // Animated dot
          if (showConn && layerProgress < 1 && layerProgress > 0) {
            const t = Math.min(1, Math.max(0, layerProgress));
            const dotX =
              phase === 'forward'
                ? src.x + nodeR + (dst.x - nodeR - src.x - nodeR) * t
                : dst.x - nodeR + (src.x + nodeR - dst.x + nodeR) * t;
            const dotY =
              phase === 'forward'
                ? src.y + (dst.y - src.y) * t
                : dst.y + (src.y - dst.y) * t;
            ctx.fillStyle =
              phase === 'forward' ? '#3b82f6' : '#ef4444';
            ctx.beginPath();
            ctx.arc(dotX, dotY, 4, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    }

    // Draw nodes
    const layerColors = ['#3b82f6', '#10b981', '#f59e0b'];
    const layerLabels = ['Input', 'Hidden (ReLU)', 'Output'];

    for (let li = 0; li < layers.length; li++) {
      for (let ni = 0; ni < layers[li]; ni++) {
        const { x, y } = positions[li][ni];

        // Glow for backward phase
        if (phase === 'backward' && animProgress > 0.3) {
          const gMag = layerGradMag[li][ni];
          const alpha = Math.min(0.4, gMag * 0.8);
          ctx.fillStyle = `rgba(239, 68, 68, ${alpha})`;
          ctx.beginPath();
          ctx.arc(x, y, nodeR + 8, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.fillStyle = layerColors[li];
        ctx.beginPath();
        ctx.arc(x, y, nodeR, 0, Math.PI * 2);
        ctx.fill();

        // Value
        ctx.fillStyle = 'white';
        ctx.font = 'bold 12px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(nodeValues[li][ni], x, y);
      }

      // Layer label
      ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(layerLabels[li], layerX[li], H - 12);
    }

    // Loss display
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 13px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText(
      `Loss = ${net.loss.toFixed(4)}`,
      W * 0.82,
      H / 2 - 10,
    );
    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
    ctx.font = '11px system-ui';
    ctx.fillText(
      `Target = ${TARGET}`,
      W * 0.82,
      H / 2 + 10,
    );

    // Phase label
    ctx.fillStyle =
      phase === 'forward' ? '#3b82f6' : '#ef4444';
    ctx.font = 'bold 13px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(
      phase === 'forward' ? 'FORWARD PASS' : 'BACKWARD PASS',
      W / 2,
      20,
    );

    // Gradient values in backward mode
    if (phase === 'backward' && animProgress > 0.5) {
      ctx.font = '10px system-ui';
      ctx.textAlign = 'center';
      for (let li = 0; li < layers.length; li++) {
        for (let ni = 0; ni < layers[li]; ni++) {
          const { x, y } = positions[li][ni];
          const g = layerGradMag[li][ni];
          ctx.fillStyle =
            g > 0.3 ? '#10b981' : g > 0.1 ? '#f59e0b' : '#ef4444';
          ctx.fillText(`g=${g.toFixed(3)}`, x, y + nodeR + 14);
        }
      }
    }
  }, [net, phase, animProgress]);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        Forward and Backward Pass
      </h3>

      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-4">
        <div>
          <label className="text-sm text-[var(--text-muted)]">
            Input x1: {x1.toFixed(2)}
          </label>
          <Slider
            min={-1}
            max={1}
            step={0.05}
            value={[x1]}
            onValueChange={([v]) => setX1(v)}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">
            Input x2: {x2.toFixed(2)}
          </label>
          <Slider
            min={-1}
            max={1}
            step={0.05}
            value={[x2]}
            onValueChange={([v]) => setX2(v)}
          />
        </div>
        <div className="flex items-end gap-2">
          <button
            onClick={() => animate('forward')}
            className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:opacity-90"
          >
            Forward
          </button>
          <button
            onClick={() => animate('backward')}
            className="rounded bg-red-600 px-4 py-1.5 text-sm font-medium text-white hover:opacity-90"
          >
            Backward
          </button>
        </div>
        <div className="text-sm text-[var(--text-muted)]">
          <div>Output: {net.output.toFixed(4)}</div>
          <div>MSE Loss: {net.loss.toFixed(4)}</div>
        </div>
      </div>

      <div ref={containerRef} className="w-full">
        <canvas
          ref={canvasRef}
          className="w-full rounded-xl bg-[var(--surface-2,#27272a)]"
        />
      </div>

      <div className="mt-3 text-xs text-[var(--text-muted)]">
        Blue nodes show values flowing forward (input to output). Red glow shows gradient magnitude flowing backward. Green = healthy gradient, amber = shrinking, red = vanishing.
      </div>
    </div>
  );
}
