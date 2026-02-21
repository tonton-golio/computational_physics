'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

export default function VarianceDecomposition() {
  const [groupSpread, setGroupSpread] = useState(3.0);
  const [withinSpread, setWithinSpread] = useState(1.0);

  const { groupLabels, groupMeans, ssb, ssw, sst, fStat, dataTraces } = useMemo(() => {
    const nGroups = 3;
    const nPerGroup = 15;
    const baseMeans = [5, 5, 5];
    const offsets = [-1, 0, 1];
    const means = baseMeans.map((m, i) => m + offsets[i] * groupSpread);
    const grandMean = means.reduce((a, b) => a + b, 0) / nGroups;

    const allData: number[][] = [];
    // Deterministic "noise" via sine pattern for reproducibility
    for (let g = 0; g < nGroups; g++) {
      const group: number[] = [];
      for (let j = 0; j < nPerGroup; j++) {
        const noise = Math.sin((g * nPerGroup + j) * 2.17 + 0.5) * withinSpread;
        group.push(means[g] + noise);
      }
      allData.push(group);
    }

    const gMeans = allData.map((g) => g.reduce((a, b) => a + b, 0) / g.length);
    let between = 0;
    let within = 0;
    for (let g = 0; g < nGroups; g++) {
      between += nPerGroup * (gMeans[g] - grandMean) ** 2;
      for (const v of allData[g]) within += (v - gMeans[g]) ** 2;
    }
    const total = between + within;
    const msb = between / (nGroups - 1);
    const msw = within / (nGroups * nPerGroup - nGroups);
    const f = msw > 0 ? msb / msw : 0;

    const labels = ['Group A', 'Group B', 'Group C'];
    const traces = allData.map((group, g) => ({
      x: group.map(() => g),
      y: group,
      type: 'scatter' as const,
      mode: 'markers' as const,
      marker: { color: ['#3b82f6', '#10b981', '#f59e0b'][g], size: 5 },
      name: labels[g],
    }));

    return {
      groupLabels: labels,
      groupMeans: gMeans,
      ssb: between, ssw: within, sst: total,
      fStat: f,
      dataTraces: traces,
    };
  }, [groupSpread, withinSpread]);

  const pctBetween = sst > 0 ? (ssb / sst * 100) : 0;
  const pctWithin = sst > 0 ? (ssw / sst * 100) : 0;

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">ANOVA: Variance Decomposition</h3>
      <div className="grid grid-cols-2 gap-6 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Between-group spread: {groupSpread.toFixed(1)}</label>
          <Slider value={[groupSpread]} onValueChange={([v]) => setGroupSpread(v)} min={0} max={5} step={0.1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Within-group spread: {withinSpread.toFixed(1)}</label>
          <Slider value={[withinSpread]} onValueChange={([v]) => setWithinSpread(v)} min={0.1} max={5} step={0.1} />
        </div>
      </div>
      <div className="mb-3 text-sm text-[var(--text-muted)]">
        F = {fStat.toFixed(2)} | SS_between = {ssb.toFixed(1)} ({pctBetween.toFixed(0)}%) | SS_within = {ssw.toFixed(1)} ({pctWithin.toFixed(0)}%)
      </div>
      <div className="grid grid-cols-2 gap-4">
        <CanvasChart
          data={dataTraces as any}
          layout={{
            height: 350,
            xaxis: { title: { text: 'Group' }, tickvals: [0, 1, 2], ticktext: groupLabels },
            yaxis: { title: { text: 'Value' } },
            shapes: groupMeans.map((m, i) => ({
              type: 'line' as const, x0: i - 0.3, x1: i + 0.3, y0: m, y1: m,
              line: { color: '#ef4444', width: 2, dash: 'dash' },
            })),
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[
            {
              x: [0, 1], y: [ssb, ssw], type: 'bar',
              marker: { color: ['#3b82f6', '#f59e0b'] },
              name: 'SS',
            },
          ]}
          layout={{
            height: 350,
            xaxis: { title: { text: '' }, tickvals: [0, 1], ticktext: ['SS Between', 'SS Within'] },
            yaxis: { title: { text: 'Sum of Squares' } },
          }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
