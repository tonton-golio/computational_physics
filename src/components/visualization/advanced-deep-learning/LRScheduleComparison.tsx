"use client";

import { useState, useMemo } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationSettings, SimulationConfig, SimulationLabel, SimulationCheckbox } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

type ScheduleType = 'step-decay' | 'cosine' | 'warmup-cosine' | 'one-cycle';

const SCHEDULE_INFO: Record<ScheduleType, { label: string; color: string; description: string }> = {
  'step-decay': {
    label: 'Step Decay',
    color: '#ef4444',
    description: 'Reduces LR by a factor at fixed milestones (epochs 30, 60, 90).',
  },
  'cosine': {
    label: 'Cosine Annealing',
    color: '#3b82f6',
    description: 'Smooth cosine curve from max LR to near zero.',
  },
  'warmup-cosine': {
    label: 'Warmup + Cosine',
    color: '#8b5cf6',
    description: 'Linear warmup followed by cosine decay. Standard for transformers.',
  },
  'one-cycle': {
    label: 'One-Cycle',
    color: '#10b981',
    description: 'Warmup to max, then cosine to min. Often fastest convergence.',
  },
};

function computeSchedule(type: ScheduleType, maxLR: number, totalEpochs: number, warmupFrac: number): number[] {
  const lrs: number[] = [];
  const minLR = maxLR * 0.01;
  const warmupEpochs = Math.floor(totalEpochs * warmupFrac);

  for (let e = 0; e < totalEpochs; e++) {
    switch (type) {
      case 'step-decay': {
        const milestones = [0.3, 0.6, 0.9].map(f => Math.floor(f * totalEpochs));
        let lr = maxLR;
        for (const m of milestones) {
          if (e >= m) lr *= 0.3;
        }
        lrs.push(lr);
        break;
      }
      case 'cosine': {
        lrs.push(minLR + 0.5 * (maxLR - minLR) * (1 + Math.cos(Math.PI * e / totalEpochs)));
        break;
      }
      case 'warmup-cosine': {
        if (e < warmupEpochs) {
          lrs.push(minLR + (maxLR - minLR) * (e / warmupEpochs));
        } else {
          const progress = (e - warmupEpochs) / (totalEpochs - warmupEpochs);
          lrs.push(minLR + 0.5 * (maxLR - minLR) * (1 + Math.cos(Math.PI * progress)));
        }
        break;
      }
      case 'one-cycle': {
        const peakEpoch = Math.floor(totalEpochs * 0.3);
        if (e < peakEpoch) {
          lrs.push(minLR + (maxLR - minLR) * (e / peakEpoch));
        } else {
          const progress = (e - peakEpoch) / (totalEpochs - peakEpoch);
          lrs.push(minLR + 0.5 * (maxLR - minLR) * (1 + Math.cos(Math.PI * progress)));
        }
        break;
      }
    }
  }
  return lrs;
}

export default function LRScheduleComparison({}: SimulationComponentProps) {
  const [selected, setSelected] = useState<Set<ScheduleType>>(
    new Set(['cosine', 'warmup-cosine', 'one-cycle'])
  );
  const [maxLR, setMaxLR] = useState(0.01);
  const [totalEpochs, setTotalEpochs] = useState(100);
  const [warmupFrac, setWarmupFrac] = useState(0.1);

  const plotData = useMemo(() => {
    const traces: any[] = [];
    const epochs = Array.from({ length: totalEpochs }, (_, i) => i);

    for (const type of ['step-decay', 'cosine', 'warmup-cosine', 'one-cycle'] as ScheduleType[]) {
      if (!selected.has(type)) continue;
      const info = SCHEDULE_INFO[type];
      const lrs = computeSchedule(type, maxLR, totalEpochs, warmupFrac);
      traces.push({
        type: 'scatter' as const,
        x: epochs,
        y: lrs,
        mode: 'lines' as const,
        line: { color: info.color, width: 2.5 },
        name: info.label,
      });
    }
    return traces;
  }, [selected, maxLR, totalEpochs, warmupFrac]);

  const toggleSchedule = (type: ScheduleType) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  return (
    <SimulationPanel title="Learning Rate Schedule Comparison">
      <SimulationSettings>
        <div className="space-y-3">
          <p className="text-sm text-[var(--text-muted)] font-semibold">Select schedules:</p>
          {(['step-decay', 'cosine', 'warmup-cosine', 'one-cycle'] as ScheduleType[]).map(type => {
            const info = SCHEDULE_INFO[type];
            return (
              <SimulationCheckbox key={type} checked={selected.has(type)} onChange={() => toggleSchedule(type)} label={info.label} style={{ color: selected.has(type) ? info.color : '#666' }} />
            );
          })}

          <div className="mt-3 p-3 bg-[var(--surface-2)] rounded text-xs text-[var(--text-muted)] space-y-2">
            {(['step-decay', 'cosine', 'warmup-cosine', 'one-cycle'] as ScheduleType[]).filter(t => selected.has(t)).map(type => (
              <div key={type}>
                <span className="font-semibold" style={{ color: SCHEDULE_INFO[type].color }}>
                  {SCHEDULE_INFO[type].label}:
                </span>{' '}
                {SCHEDULE_INFO[type].description}
              </div>
            ))}
          </div>
        </div>
      </SimulationSettings>
      <SimulationConfig>
        <div className="space-y-3">
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Max LR: {maxLR}</SimulationLabel>
            <Slider min={0.001} max={0.1} step={0.001} value={[maxLR]} onValueChange={([v]) => setMaxLR(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Total epochs: {totalEpochs}</SimulationLabel>
            <Slider min={20} max={200} step={10} value={[totalEpochs]} onValueChange={([v]) => setTotalEpochs(v)} className="w-full" />
          </div>
          <div>
            <SimulationLabel className="block text-sm text-[var(--text-muted)] mb-1">Warmup fraction: {warmupFrac.toFixed(2)}</SimulationLabel>
            <Slider min={0} max={0.3} step={0.01} value={[warmupFrac]} onValueChange={([v]) => setWarmupFrac(v)} className="w-full" />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
        <CanvasChart
          data={plotData}
          layout={{
            xaxis: { title: { text: 'Epoch' } },
            yaxis: { title: { text: 'Learning Rate' } },
            margin: { t: 30, b: 50, l: 70, r: 30 },
            autosize: true,
          }}
          style={{ width: '100%', height: '400px' }}
        />
      </SimulationMain>
    </SimulationPanel>
  );
}
