'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart, type ChartTrace } from '@/components/ui/canvas-chart';

/**
 * Compare Newtonian (parabolic) vs power-law fluid velocity profiles in pipe
 * flow. The power-law index n controls the shape: n=1 is Newtonian, n<1 is
 * shear-thinning (blunted profile), n>1 is shear-thickening (pointed profile).
 * The glacier case (n~3) is highlighted.
 */

export default function PoiseuilleVsPowerLaw() {
  const [n, setN] = useState(1.0);
  const [showGlacier, setShowGlacier] = useState(false);

  const data = useMemo(() => {
    const N = 200;
    const a = 1; // pipe radius normalised
    // Generalised Poiseuille for power-law fluid:
    // v(r) = v_max * (1 - (r/a)^((n+1)/n))
    // For Newtonian (n=1): v = v_max * (1 - r^2/a^2) — standard parabola

    const rArr: number[] = [];
    const vNewtonArr: number[] = [];
    const vPowerArr: number[] = [];
    const vGlacierArr: number[] = [];

    for (let i = 0; i <= N; i++) {
      const r = -a + (2 * a * i) / N; // -a to a for symmetric profile
      const rAbs = Math.abs(r);
      rArr.push(r);

      // Newtonian (n=1)
      vNewtonArr.push(1 - (rAbs / a) ** 2);

      // Power-law
      const exp = (n + 1) / n;
      vPowerArr.push(1 - (rAbs / a) ** exp);

      // Glacier (n=3)
      if (showGlacier) {
        const expG = (3 + 1) / 3;
        vGlacierArr.push(1 - (rAbs / a) ** expG);
      }
    }

    const traces: ChartTrace[] = [
      {
        type: 'scatter',
        mode: 'lines',
        x: vNewtonArr,
        y: rArr,
        name: 'Newtonian (n=1)',
        line: { color: '#8fa3c4', width: 2, dash: 'dash' },
      },
      {
        type: 'scatter',
        mode: 'lines',
        x: vPowerArr,
        y: rArr,
        name: `Power-law (n=${n.toFixed(1)})`,
        line: { color: '#3b82f6', width: 2.5 },
      },
    ];

    if (showGlacier && Math.abs(n - 3) > 0.05) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: vGlacierArr,
        y: rArr,
        name: 'Glen\'s law (n=3)',
        line: { color: '#10b981', width: 2, dash: 'dot' },
      });
    }

    return traces;
  }, [n, showGlacier]);

  const layout = useMemo(() => ({
    xaxis: { title: { text: 'v / v_max' }, range: [0, 1.05] },
    yaxis: { title: { text: 'r / a' }, range: [-1.1, 1.1] },
  }), []);

  const regime = n < 1 ? 'shear-thinning (blunted)' : n > 1 ? 'shear-thickening (pointed)' : 'Newtonian (parabolic)';

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-2 text-[var(--text-strong)]">
        Poiseuille Flow: Newtonian vs Power-Law Fluid
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        The velocity profile across a pipe depends on the power-law index n.
        Shear-thinning fluids (n&lt;1) have a blunted profile; shear-thickening
        fluids (n&gt;1) have a more pointed profile. Glaciers use n&asymp;3.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">
            Power-law index n: {n.toFixed(1)} &mdash; {regime}
          </label>
          <Slider min={0.2} max={5} step={0.1} value={[n]}
            onValueChange={([v]) => setN(v)} className="w-full" />
        </div>
        <div className="flex items-end">
          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showGlacier}
              onChange={(e) => setShowGlacier(e.target.checked)}
              className="rounded"
            />
            Show Glen&apos;s flow law (n=3)
          </label>
        </div>
      </div>
      <CanvasChart data={data} layout={layout} style={{ width: '100%', height: 420 }} />
      <p className="mt-3 text-xs text-[var(--text-muted)]">
        Dashed grey: Newtonian parabola. Blue: current power-law profile.
        Green dotted: glacier ice (n=3). Notice how n&gt;1 makes the profile
        flatter in the centre &mdash; most of the shearing happens near the walls.
      </p>
    </div>
  );
}
