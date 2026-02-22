"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

/**
 * Demonstrate aliasing in action. A high-frequency sine wave sampled at
 * different rates. When below Nyquist, the aliased low-frequency wave appears.
 */
export default function AliasingSlider({}: SimulationComponentProps) {
  const [signalFreq, setSignalFreq] = useState(8);    // Hz
  const [sampleRate, setSampleRate] = useState(20);     // Hz
  const [duration] = useState(1.0);                     // seconds

  const nyquistFreq = sampleRate / 2;
  const isAliased = signalFreq > nyquistFreq;

  // The apparent (aliased) frequency
  const apparentFreq = useMemo(() => {
    if (!isAliased) return signalFreq;
    // Aliased frequency = |signalFreq - round(signalFreq/sampleRate)*sampleRate|
    const k = Math.round(signalFreq / sampleRate);
    return Math.abs(signalFreq - k * sampleRate);
  }, [signalFreq, sampleRate, isAliased]);

  const data = useMemo(() => {
    // True continuous signal (dense samples for smooth curve)
    const nContinuous = 1000;
    const tCont: number[] = [];
    const yCont: number[] = [];
    for (let i = 0; i < nContinuous; i++) {
      const t = (duration * i) / (nContinuous - 1);
      tCont.push(t);
      yCont.push(Math.sin(2 * Math.PI * signalFreq * t));
    }

    // Sampled points
    const nSamples = Math.floor(sampleRate * duration);
    const tSamp: number[] = [];
    const ySamp: number[] = [];
    for (let i = 0; i <= nSamples; i++) {
      const t = i / sampleRate;
      if (t > duration) break;
      tSamp.push(t);
      ySamp.push(Math.sin(2 * Math.PI * signalFreq * t));
    }

    // Reconstructed signal from samples (sinc interpolation approximation:
    // just show the apparent frequency sine wave)
    const tRecon: number[] = [];
    const yRecon: number[] = [];
    for (let i = 0; i < nContinuous; i++) {
      const t = (duration * i) / (nContinuous - 1);
      tRecon.push(t);
      yRecon.push(Math.sin(2 * Math.PI * apparentFreq * t));
    }

    return { tCont, yCont, tSamp, ySamp, tRecon, yRecon };
  }, [signalFreq, sampleRate, duration, apparentFreq]);

  // Power spectrum visualization
  const spectrumData = useMemo(() => {
    const maxFreq = 50;
    const nFreqs = 500;
    const freqs: number[] = [];
    const power: number[] = [];
    const nyqLine: number[] = [];

    for (let i = 0; i < nFreqs; i++) {
      const f = (maxFreq * i) / (nFreqs - 1);
      freqs.push(f);

      // Simple peak at signal frequency and its aliases
      let p = 0;
      for (let k = -3; k <= 3; k++) {
        const aliasF = Math.abs(signalFreq - k * sampleRate);
        const dist = f - aliasF;
        p += Math.exp(-(dist * dist) / 0.3);
      }
      power.push(p);
      nyqLine.push(0);
    }

    return { freqs, power, nyqLine };
  }, [signalFreq, sampleRate]);

  return (
    <SimulationPanel title="Aliasing and the Nyquist Frequency" caption="When the sampling rate is too low, high frequencies masquerade as low ones. Increase the signal frequency past the Nyquist limit and watch the aliased wave appear.">
      <SimulationConfig>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <SimulationLabel>Signal frequency: {signalFreq.toFixed(1)} Hz</SimulationLabel>
            <Slider
              value={[signalFreq]}
              onValueChange={([v]) => setSignalFreq(v)}
              min={1}
              max={40}
              step={0.5}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>Sampling rate: {sampleRate.toFixed(0)} Hz (Nyquist: {nyquistFreq.toFixed(1)} Hz)</SimulationLabel>
            <Slider
              value={[sampleRate]}
              onValueChange={([v]) => setSampleRate(v)}
              min={4}
              max={80}
              step={1}
              className="w-full"
            />
          </div>
        </div>

        {isAliased && (
          <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
            Aliasing detected! Signal at {signalFreq.toFixed(1)} Hz exceeds Nyquist frequency of{' '}
            {nyquistFreq.toFixed(1)} Hz. The apparent frequency is {apparentFreq.toFixed(1)} Hz.
          </div>
        )}
      </SimulationConfig>

      <SimulationMain>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <CanvasChart
            data={[
              {
                x: data.tCont,
                y: data.yCont,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#3b82f6', width: 1.5 },
                name: `True signal (${signalFreq} Hz)`,
              },
              ...(isAliased
                ? [
                    {
                      x: data.tRecon,
                      y: data.yRecon,
                      type: 'scatter' as const,
                      mode: 'lines' as const,
                      line: { color: '#ef4444', width: 2, dash: 'dash' as const },
                      name: `Aliased (${apparentFreq.toFixed(1)} Hz)`,
                    },
                  ]
                : []),
              {
                x: data.tSamp,
                y: data.ySamp,
                type: 'scatter',
                mode: 'markers',
                marker: { color: '#10b981', size: 6 },
                name: `Samples (${sampleRate} Hz)`,
              },
            ]}
            layout={{
              title: { text: 'Time Domain' },
              xaxis: { title: { text: 'Time (s)' } },
              yaxis: { title: { text: 'Amplitude' }, range: [-1.3, 1.3] },
              showlegend: true,
              margin: { t: 40, r: 20, b: 50, l: 50 },
            }}
            style={{ width: '100%', height: 380 }}
          />

          <CanvasChart
            data={[
              {
                x: spectrumData.freqs,
                y: spectrumData.power,
                type: 'scatter',
                mode: 'lines',
                line: { color: '#f59e0b', width: 2 },
                fill: 'tozeroy',
                fillcolor: 'rgba(245,158,11,0.15)',
                name: 'Spectrum',
              },
            ]}
            layout={{
              title: { text: 'Frequency Domain (with aliases)' },
              xaxis: { title: { text: 'Frequency (Hz)' } },
              yaxis: { title: { text: 'Power' } },
              shapes: [
                {
                  type: 'line',
                  x0: nyquistFreq,
                  x1: nyquistFreq,
                  y0: 'min' as unknown as number,
                  y1: 'max' as unknown as number,
                  line: { color: '#ef4444', width: 2, dash: 'dash' },
                },
              ],
              showlegend: true,
              margin: { t: 40, r: 20, b: 50, l: 50 },
            }}
            style={{ width: '100%', height: 380 }}
          />
        </div>
      </SimulationMain>

      <SimulationResults>
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Signal Freq</div>
            <div className="text-sm font-mono font-semibold text-[#3b82f6]">
              {signalFreq.toFixed(1)} Hz
            </div>
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
            <div className="text-xs text-[var(--text-muted)]">Nyquist Freq</div>
            <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
              {nyquistFreq.toFixed(1)} Hz
            </div>
          </div>
          <div className={`rounded-md border p-2.5 text-center ${isAliased ? 'border-red-500/50 bg-red-500/10' : 'border-[var(--border)] bg-[var(--surface-2)]/50'}`}>
            <div className="text-xs text-[var(--text-muted)]">Apparent Freq</div>
            <div className={`text-sm font-mono font-semibold ${isAliased ? 'text-red-400' : 'text-[var(--accent)]'}`}>
              {apparentFreq.toFixed(1)} Hz
            </div>
          </div>
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
