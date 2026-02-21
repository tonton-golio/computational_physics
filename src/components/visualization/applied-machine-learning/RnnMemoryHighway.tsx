'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Slider } from '@/components/ui/slider';

const DEFAULT_SENTENCE =
  'The author who grew up in Paris and studied mathematics at the Sorbonne finally published her long-awaited book about topology';

/**
 * Simulate hidden state decay for a Simple RNN.
 * At each timestep the hidden state decays by a factor (simulating vanishing gradients).
 * The initial "signal" is injected at the source word position.
 */
function simulateRNN(
  nSteps: number,
  sourceIdx: number,
  decayFactor: number,
): number[] {
  const states = new Array(nSteps).fill(0);
  states[sourceIdx] = 1.0;
  for (let t = sourceIdx + 1; t < nSteps; t++) {
    states[t] = states[t - 1] * decayFactor;
  }
  // Backward pass from source
  for (let t = sourceIdx - 1; t >= 0; t--) {
    states[t] = states[t + 1] * decayFactor;
  }
  return states;
}

/**
 * Simulate LSTM cell state: the "highway" preserves information much longer.
 * Forget gate is close to 1 (e.g. 0.95-0.99), so information persists.
 */
function simulateLSTM(
  nSteps: number,
  sourceIdx: number,
  forgetGate: number,
): number[] {
  const states = new Array(nSteps).fill(0);
  states[sourceIdx] = 1.0;
  for (let t = sourceIdx + 1; t < nSteps; t++) {
    states[t] = states[t - 1] * forgetGate;
  }
  for (let t = sourceIdx - 1; t >= 0; t--) {
    states[t] = states[t + 1] * forgetGate;
  }
  return states;
}

export default function RnnMemoryHighway(): React.ReactElement {
  const rnnCanvasRef = useRef<HTMLCanvasElement>(null);
  const lstmCanvasRef = useRef<HTMLCanvasElement>(null);
  const [sentence, setSentence] = useState(DEFAULT_SENTENCE);
  const [highlightIdx, setHighlightIdx] = useState(0);
  const [nSteps, setNSteps] = useState(30);
  const [rnnDecay, setRnnDecay] = useState(0.7);

  const words = useMemo(() => {
    const split = sentence.split(/\s+/);
    return split.length > nSteps ? split.slice(0, nSteps) : split;
  }, [sentence, nSteps]);

  const effectiveSteps = Math.min(words.length, nSteps);

  const rnnStates = useMemo(
    () => simulateRNN(effectiveSteps, highlightIdx, rnnDecay),
    [effectiveSteps, highlightIdx, rnnDecay],
  );

  const lstmStates = useMemo(
    () => simulateLSTM(effectiveSteps, highlightIdx, 0.97),
    [effectiveSteps, highlightIdx],
  );

  const drawTimeline = useCallback(
    (
      canvas: HTMLCanvasElement | null,
      states: number[],
      color: string,
      label: string,
      showHighway: boolean,
    ) => {
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const n = states.length;
      const cellW = Math.min(50, (W - 40) / n);
      const startX = 20;
      const cellH = 40;
      const cellY = H / 2 - cellH / 2;

      // Highway line for LSTM
      if (showHighway) {
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 6;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(startX, cellY + cellH / 2);
        ctx.lineTo(startX + n * cellW, cellY + cellH / 2);
        ctx.stroke();
      }

      // Draw cells
      for (let i = 0; i < n; i++) {
        const x = startX + i * cellW;
        const intensity = states[i];

        // Cell background with intensity-based opacity
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${Math.max(0.05, intensity)})`;
        ctx.fillRect(x + 1, cellY, cellW - 2, cellH);

        // Border
        ctx.strokeStyle = i === highlightIdx ? '#fbbf24' : 'rgba(148, 163, 184, 0.3)';
        ctx.lineWidth = i === highlightIdx ? 2 : 1;
        ctx.setLineDash([]);
        ctx.strokeRect(x + 1, cellY, cellW - 2, cellH);

        // Word label (rotated for narrow cells)
        if (cellW > 20 && i < words.length) {
          ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
          ctx.font = `${Math.min(10, cellW * 0.25)}px system-ui`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          const word = words[i].length > 6 ? words[i].slice(0, 5) + '.' : words[i];
          ctx.fillText(word, x + cellW / 2, cellY + cellH + 4);
        }

        // Intensity value
        if (cellW > 25) {
          ctx.fillStyle = intensity > 0.5 ? 'white' : 'rgba(226, 232, 240, 0.6)';
          ctx.font = `${Math.min(10, cellW * 0.3)}px system-ui`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(intensity.toFixed(2), x + cellW / 2, cellY + cellH / 2);
        }
      }

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 13px system-ui';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(label, startX, 6);
    },
    [highlightIdx, words],
  );

  useEffect(() => {
    drawTimeline(rnnCanvasRef.current, rnnStates, '#ef4444', 'Simple RNN (vanishing)', false);
    drawTimeline(lstmCanvasRef.current, lstmStates, '#10b981', 'LSTM (cell-state highway)', true);
  }, [drawTimeline, rnnStates, lstmStates]);

  return (
    <div className="w-full rounded-lg bg-[var(--surface-1)] p-6">
      <h3 className="mb-4 text-xl font-semibold text-[var(--text-strong)]">
        RNN vs LSTM: Memory Highway
      </h3>

      <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <label className="text-sm text-[var(--text-muted)]">Highlight word: {highlightIdx}</label>
          <Slider
            min={0}
            max={Math.max(0, effectiveSteps - 1)}
            step={1}
            value={[highlightIdx]}
            onValueChange={([v]) => setHighlightIdx(v)}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">RNN decay: {rnnDecay.toFixed(2)}</label>
          <Slider
            min={0.3}
            max={0.95}
            step={0.01}
            value={[rnnDecay]}
            onValueChange={([v]) => setRnnDecay(v)}
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)]">Timesteps: {nSteps}</label>
          <Slider
            min={10}
            max={60}
            step={1}
            value={[nSteps]}
            onValueChange={([v]) => setNSteps(v)}
          />
        </div>
      </div>

      <div className="mb-3">
        <input
          type="text"
          value={sentence}
          onChange={(e) => setSentence(e.target.value)}
          className="w-full rounded bg-[var(--surface-2,#27272a)] px-3 py-2 text-sm text-[var(--text-strong)]"
          placeholder="Type a sentence..."
        />
      </div>

      <div className="space-y-4">
        <canvas
          ref={rnnCanvasRef}
          width={700}
          height={120}
          className="w-full rounded-lg bg-[var(--surface-2,#27272a)]"
          style={{ maxWidth: 700 }}
        />
        <canvas
          ref={lstmCanvasRef}
          width={700}
          height={120}
          className="w-full rounded-lg bg-[var(--surface-2,#27272a)]"
          style={{ maxWidth: 700 }}
        />
      </div>

      <div className="mt-3 rounded bg-[var(--surface-2,#27272a)] p-3 text-sm text-[var(--text-muted)]">
        Move the highlight slider to pick a word. Watch how the Simple RNN&apos;s
        memory (red) fades rapidly with distance, while the LSTM&apos;s cell-state
        highway (green) preserves information across long sequences. The LSTM&apos;s
        forget gate stays near 1.0, allowing gradients to flow without vanishing.
      </div>
    </div>
  );
}
