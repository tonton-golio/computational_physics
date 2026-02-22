"use client";

import { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationPanel, SimulationConfig, SimulationResults, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import type { SimulationComponentProps } from '@/shared/types/simulation';

export default function RandomizationVsBlocking({}: SimulationComponentProps) {
  const [blockEffect, setBlockEffect] = useState(3.0);
  const [treatmentEffect, setTreatmentEffect] = useState(2.0);

  const { crdResiduals, rbdResiduals, crdSE, rbdSE } = useMemo(() => {
    const nBlocks = 4;
    const nTreat = 3;
    const nTotal = nBlocks * nTreat;
    const blockMeans = [0, 1, 2, 3].map((b) => b * blockEffect);
    const treatMeans = [0, treatmentEffect, treatmentEffect * 2];

    // Generate data with deterministic noise
    const crdRes: number[] = [];
    const rbdRes: number[] = [];
    const crdX: number[] = [];
    const rbdX: number[] = [];

    // CRD: block effect is NOT removed from error
    let crdSS = 0;
    let rbdSS = 0;
    for (let b = 0; b < nBlocks; b++) {
      for (let t = 0; t < nTreat; t++) {
        const idx = b * nTreat + t;
        const noise = Math.sin(idx * 3.14 + 0.7) * 1.5;
        const y = treatMeans[t] + blockMeans[b] + noise;
        const groupMean = treatMeans[t] + (blockMeans.reduce((a, v) => a + v, 0) / nBlocks);
        // In CRD, the block effect is part of the residual
        const crdResidual = y - groupMean;
        // In RBD, the block effect is removed
        const rbdResidual = y - groupMean - blockMeans[b];
        crdRes.push(crdResidual);
        rbdRes.push(rbdResidual);
        crdX.push(idx);
        rbdX.push(idx);
        crdSS += crdResidual * crdResidual;
        rbdSS += rbdResidual * rbdResidual;
      }
    }

    const crdMSE = crdSS / (nTotal - nTreat);
    const rbdMSE = rbdSS / ((nTotal - nTreat) - (nBlocks - 1));
    const crdSEval = Math.sqrt(crdMSE / (nTotal / nTreat));
    const rbdSEval = Math.sqrt(rbdMSE / (nTotal / nTreat));

    return {
      crdResiduals: { x: crdX, y: crdRes },
      rbdResiduals: { x: rbdX, y: rbdRes },
      crdSE: crdSEval,
      rbdSE: rbdSEval,
    };
  }, [blockEffect, treatmentEffect]);

  return (
    <SimulationPanel title="Randomization vs Blocking">
      <SimulationConfig>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <SimulationLabel>Block effect strength: {blockEffect.toFixed(1)}</SimulationLabel>
            <Slider value={[blockEffect]} onValueChange={([v]) => setBlockEffect(v)} min={0} max={6} step={0.1} />
          </div>
          <div>
            <SimulationLabel>Treatment effect: {treatmentEffect.toFixed(1)}</SimulationLabel>
            <Slider value={[treatmentEffect]} onValueChange={([v]) => setTreatmentEffect(v)} min={0} max={5} step={0.1} />
          </div>
        </div>
      </SimulationConfig>
      <SimulationMain>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-center text-[var(--text-muted)] mb-1">CRD Residuals (block noise included)</p>
          <CanvasChart
            data={[
              {
                x: crdResiduals.x, y: crdResiduals.y, type: 'scatter', mode: 'markers',
                marker: { color: '#ef4444', size: 5 }, name: 'CRD residuals',
              },
            ]}
            layout={{
              height: 280,
              xaxis: { title: { text: 'Observation' } },
              yaxis: { title: { text: 'Residual' } },
              shapes: [{ type: 'line', x0: 0, x1: 11, y0: 0, y1: 0, line: { color: '#94a3b8', width: 1, dash: 'dash' } }],
            }}
            style={{ width: '100%' }}
          />
        </div>
        <div>
          <p className="text-sm text-center text-[var(--text-muted)] mb-1">RBD Residuals (block effect removed)</p>
          <CanvasChart
            data={[
              {
                x: rbdResiduals.x, y: rbdResiduals.y, type: 'scatter', mode: 'markers',
                marker: { color: '#10b981', size: 5 }, name: 'RBD residuals',
              },
            ]}
            layout={{
              height: 280,
              xaxis: { title: { text: 'Observation' } },
              yaxis: { title: { text: 'Residual' } },
              shapes: [{ type: 'line', x0: 0, x1: 11, y0: 0, y1: 0, line: { color: '#94a3b8', width: 1, dash: 'dash' } }],
            }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
      </SimulationMain>
      <SimulationResults>
        <div className="text-sm text-[var(--text-muted)]">
          SE(CRD) = {crdSE.toFixed(3)} | SE(RBD) = {rbdSE.toFixed(3)} |
          Relative efficiency: {crdSE > 0 ? ((crdSE / rbdSE) ** 2).toFixed(2) : 'N/A'}x
        </div>
      </SimulationResults>
    </SimulationPanel>
  );
}
