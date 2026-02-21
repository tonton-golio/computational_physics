'use client';

import { useState, useMemo, useRef, useEffect } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';

const GRID = 21, CELL = 16;

function rfSize(layers: number, kernel: number) { return 1 + layers * (kernel - 1); }

export default function ReceptiveFieldGrowth() {
  const [numLayers, setNumLayers] = useState(3);
  const [kernelSize, setKernelSize] = useState(3);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const rf = useMemo(() => rfSize(numLayers, kernelSize), [numLayers, kernelSize]);

  const chartData: ChartTrace[] = useMemo(() => {
    const layers = Array.from({ length: 10 }, (_, i) => i + 1);
    return [3, 5, 7].map(ks => ({
      x: layers, y: layers.map(l => rfSize(l, ks)),
      type: 'scatter' as const, mode: 'lines+markers' as const,
      marker: { size: 4 },
      line: { color: ks === 3 ? '#3b82f6' : ks === 5 ? '#10b981' : '#f59e0b',
        width: ks === kernelSize ? 3 : 1.5, dash: (ks === kernelSize ? 'solid' : 'dash') as 'solid' | 'dash' },
      name: `${ks}x${ks} kernel`,
    }));
  }, [kernelSize]);

  useEffect(() => {
    const canvas = canvasRef.current, container = containerRef.current;
    if (!canvas || !container) return;
    const redraw = () => {
      const total = GRID * CELL + 2, dpr = devicePixelRatio || 1;
      canvas.width = total * dpr; canvas.height = total * dpr;
      canvas.style.width = `${total}px`; canvas.style.height = `${total}px`;
      const ctx = canvas.getContext('2d'); if (!ctx) return;
      ctx.scale(dpr, dpr); ctx.clearRect(0, 0, total, total);
      const ctr = GRID >> 1, halfRF = rf >> 1;

      for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) {
        const x = c * CELL + 1, y = r * CELL + 1;
        const dr = Math.abs(r - ctr), dc = Math.abs(c - ctr);
        if (r === ctr && c === ctr) ctx.fillStyle = '#ef4444';
        else if (dr <= halfRF && dc <= halfRF) {
          const dist = Math.max(dr, dc) / (halfRF || 1);
          const layer = Math.ceil(Math.max(dr, dc) / Math.max(kernelSize - 1, 1));
          ctx.fillStyle = `hsla(${210 + layer * 30}, 70%, 55%, ${Math.max(0.15, 1 - dist * 0.7)})`;
        } else ctx.fillStyle = 'rgba(100,100,100,0.15)';
        ctx.fillRect(x, y, CELL - 1, CELL - 1);
        ctx.strokeStyle = 'rgba(150,150,150,0.3)'; ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, CELL - 1, CELL - 1);
      }

      // RF outline
      const rfPx = Math.min(rf, GRID);
      const s = (ctr - (rfPx >> 1)) * CELL + 1;
      ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 2;
      ctx.strokeRect(s, s, rfPx * CELL, rfPx * CELL);

      // Layer rings
      for (let l = 1; l <= numLayers; l++) {
        const lrf = Math.min(rfSize(l, kernelSize), GRID), lh = lrf >> 1;
        const lp = (ctr - lh) * CELL + 1;
        ctx.strokeStyle = `hsla(${210 + l * 30}, 70%, 55%, 0.6)`; ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]); ctx.strokeRect(lp, lp, lrf * CELL, lrf * CELL); ctx.setLineDash([]);
      }
    };
    redraw();
    const ro = new ResizeObserver(redraw); ro.observe(container);
    return () => ro.disconnect();
  }, [numLayers, kernelSize, rf]);

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Receptive Field Growth</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">Conv layers: {numLayers}</label>
          <Slider min={1} max={8} step={1} value={[numLayers]} onValueChange={([v]) => setNumLayers(v)} />
        </div>
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">Kernel: {kernelSize}x{kernelSize}</label>
          <Slider min={3} max={7} step={2} value={[kernelSize]} onValueChange={([v]) => setKernelSize(v)} />
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        <SimulationMain scaleMode="contain" className="flex flex-col items-center">
          <p className="text-sm text-[var(--text-muted)] mb-2 font-semibold">
            Input grid — RF = {rf}x{rf}{rf > GRID ? ` (exceeds ${GRID}x${GRID} view)` : ''}
          </p>
          <div ref={containerRef}><canvas ref={canvasRef} style={{ display: 'block', borderRadius: '4px' }} /></div>
          <div className="flex gap-4 mt-3 text-xs text-[var(--text-muted)]">
            <span><span className="inline-block w-3 h-3 rounded-sm bg-red-500 mr-1 align-middle" /> Neuron</span>
            <span><span className="inline-block w-3 h-3 rounded-sm border-2 border-blue-500 mr-1 align-middle" /> RF boundary</span>
          </div>
        </SimulationMain>
        <CanvasChart data={chartData} layout={{ title: { text: 'RF size vs. depth' },
          xaxis: { title: { text: 'Number of layers' } }, yaxis: { title: { text: 'Receptive field (px)' } },
          margin: { t: 40, b: 50, l: 60, r: 20 } }} style={{ width: '100%', height: 370 }} />
      </div>
      <div className="mt-4 p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        Each {kernelSize}x{kernelSize} conv layer expands the receptive field by {kernelSize - 1} pixels per
        direction. After {numLayers} layer{numLayers > 1 ? 's' : ''}, one neuron sees a {rf}x{rf} input
        region. Dashed rings show the cumulative RF at each depth.
      </div>
    </div>
  );
}
