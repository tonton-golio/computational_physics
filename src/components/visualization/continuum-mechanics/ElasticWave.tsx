"use client";

import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationResults, SimulationLabel, SimulationPlayButton, SimulationButton } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';


export default function ElasticWave({}: SimulationComponentProps) {
  const [youngsMod, setYoungsMod] = useState(200); // GPa
  const [density, setDensity] = useState(7800); // kg/m^3 (steel-like)
  const [amplitude, setAmplitude] = useState(1.0);
  const [frequency, setFrequency] = useState(2.0); // Hz (scaled for visualization)
  const [damping, setDamping] = useState(0.0); // damping coefficient
  const [time, setTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const animRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  // Wave speed c = sqrt(E / rho)
  // For visualization, we scale units: E in GPa = 1e9 Pa, rho in kg/m^3
  // c = sqrt(E * 1e9 / rho) in m/s, but we normalize for display
  const waveSpeed = useMemo(() => {
    return Math.sqrt((youngsMod * 1e9) / density);
  }, [youngsMod, density]);

  // Normalized wave speed for visualization (we display a bar of length L=1)
  const cNorm = useMemo(() => {
    // scale so wave crosses bar in a reasonable number of time steps
    // reference: steel ~ 5000 m/s, we normalize so reference gives cNorm ~ 0.5
    return waveSpeed / 10000;
  }, [waveSpeed]);

  const animate = useCallback(
    (timestamp: number) => {
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp;
      }
      const dt = (timestamp - lastTimeRef.current) / 1000; // seconds
      lastTimeRef.current = timestamp;
      setTime((prev) => prev + dt * 0.5); // slow factor
      animRef.current = requestAnimationFrame(animate);
    },
    []
  );

  useEffect(() => {
    if (isPlaying) {
      lastTimeRef.current = 0;
      animRef.current = requestAnimationFrame(animate);
    } else if (animRef.current !== null) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
    return () => {
      if (animRef.current !== null) {
        cancelAnimationFrame(animRef.current);
      }
    };
  }, [isPlaying, animate]);

  const plotData = useMemo(() => {
    const N = 400; // spatial points
    const L = 1.0; // bar length (normalized)
    const omega = 2 * Math.PI * frequency;
    const k = omega / cNorm; // wave number
    const alpha = damping;

    const x: number[] = [];
    const uForward: number[] = [];
    const uReflected: number[] = [];
    const uTotal: number[] = [];

    for (let i = 0; i <= N; i++) {
      const xi = (i / N) * L;
      x.push(xi);

      // Forward (incident) wave: A * exp(-alpha*x) * sin(kx - omega*t)
      const envFwd = amplitude * Math.exp(-alpha * xi);
      const fwd = envFwd * Math.sin(k * xi - omega * time);

      // Reflected wave from fixed end at x = L:
      // Reflection from a fixed end inverts the wave
      const envRef = amplitude * Math.exp(-alpha * (2 * L - xi));
      const ref = -envRef * Math.sin(k * (2 * L - xi) - omega * time);

      uForward.push(fwd);
      uReflected.push(ref);
      uTotal.push(fwd + ref);
    }

    return {
      data: [
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x,
          y: uTotal,
          name: 'Total displacement',
          line: { color: '#3b82f6', width: 2.5 },
        },
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x,
          y: uForward,
          name: 'Incident wave',
          line: { color: '#10b981', width: 1.5, dash: 'dash' as const },
        },
        {
          type: 'scatter' as const,
          mode: 'lines' as const,
          x,
          y: uReflected,
          name: 'Reflected wave',
          line: { color: '#ef4444', width: 1.5, dash: 'dot' as const },
        },
      ],
      layout: ({
        title: {
          text: '1D Elastic Wave Propagation in a Bar',
        },
        xaxis: {
          title: { text: 'Position along bar (x / L)' },
          range: [0, L],
        },
        yaxis: {
          title: { text: 'Displacement u(x, t)' },
          range: [-2.5 * amplitude, 2.5 * amplitude],
        },
        height: 450,
        legend: {
          bgcolor: 'rgba(0,0,0,0)',
        },
        margin: { t: 50, b: 60, l: 70, r: 30 },
        shapes: [
          // Fixed end markers
          {
            type: 'line' as const,
            x0: 0,
            x1: 0,
            y0: -2.5 * amplitude,
            y1: 2.5 * amplitude,
            line: { color: '#f59e0b', width: 3 },
          },
          {
            type: 'line' as const,
            x0: 1,
            x1: 1,
            y0: -2.5 * amplitude,
            y1: 2.5 * amplitude,
            line: { color: '#f59e0b', width: 3 },
          },
        ],
      }),
    };
  }, [amplitude, frequency, damping, time, cNorm]);

  return (
    <SimulationPanel title="1D Elastic Wave Propagation" caption="Visualize longitudinal elastic wave propagation in a 1D bar with fixed boundaries. The incident wave reflects at the far end with phase inversion (fixed boundary condition), creating standing-wave patterns.">
      <SimulationSettings>
        <SimulationPlayButton isRunning={isPlaying} onToggle={() => setIsPlaying(!isPlaying)} />
        <SimulationButton
          onClick={() => {
            setIsPlaying(false);
            setTime(0);
          }}
        >
          Reset
        </SimulationButton>
        {!isPlaying && (
          <div className="flex items-center gap-2">
            <SimulationLabel>Time:</SimulationLabel>
            <Slider
              min={0}
              max={10}
              step={0.05}
              value={[time]}
              onValueChange={([v]) => setTime(v)}
              className="w-48"
            />
            <span className="text-sm font-mono text-[var(--text-muted)]">
              {time.toFixed(2)}
            </span>
          </div>
        )}
      </SimulationSettings>
      <SimulationConfig>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <SimulationLabel>
              Young&apos;s Modulus E: {youngsMod} GPa
            </SimulationLabel>
            <Slider
              min={10}
              max={400}
              step={10}
              value={[youngsMod]}
              onValueChange={([v]) => setYoungsMod(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>
              Density &rho;: {density} kg/m&sup3;
            </SimulationLabel>
            <Slider
              min={1000}
              max={20000}
              step={100}
              value={[density]}
              onValueChange={([v]) => setDensity(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>
              Amplitude A: {amplitude.toFixed(1)}
            </SimulationLabel>
            <Slider
              min={0.1}
              max={2.0}
              step={0.1}
              value={[amplitude]}
              onValueChange={([v]) => setAmplitude(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>
              Frequency f: {frequency.toFixed(1)} (normalized)
            </SimulationLabel>
            <Slider
              min={0.5}
              max={8.0}
              step={0.5}
              value={[frequency]}
              onValueChange={([v]) => setFrequency(v)}
              className="w-full"
            />
          </div>
          <div>
            <SimulationLabel>
              Damping &alpha;: {damping.toFixed(1)}
            </SimulationLabel>
            <Slider
              min={0}
              max={5}
              step={0.1}
              value={[damping]}
              onValueChange={([v]) => setDamping(v)}
              className="w-full"
            />
          </div>
        </div>
      </SimulationConfig>

      <SimulationMain>
        <CanvasChart
          data={plotData.data}
          layout={plotData.layout}
          style={{ width: '100%', height: 450 }}
        />
      </SimulationMain>
      <SimulationResults>
        <div className="bg-[var(--surface-2)] rounded-lg p-3 inline-block">
          <span className="text-[var(--text-muted)] text-sm">
            Wave speed c = &radic;(E / &rho;) ={' '}
            <span className="text-blue-400 font-mono">
              {waveSpeed.toFixed(0)} m/s
            </span>
          </span>
        </div>
      </SimulationResults>

      <div className="mt-4 text-sm text-[var(--text-muted)] space-y-1">
        <p>
          <strong className="text-[var(--text-muted)]">Blue solid:</strong> Superposition of
          incident and reflected waves (what you would physically observe).
        </p>
        <p>
          <strong className="text-[var(--text-muted)]">Green dashed:</strong> Incident wave
          traveling to the right. <strong className="text-[var(--text-muted)]">Red dotted:</strong>{' '}
          Reflected wave traveling to the left with inverted phase.
        </p>
        <p>
          <strong className="text-[var(--text-muted)]">Orange bars:</strong> Fixed boundary
          conditions at x = 0 and x = L. Increasing damping attenuates the wave
          exponentially with distance.
        </p>
      </div>
    </SimulationPanel>
  );
}
