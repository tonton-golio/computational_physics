'use client';

import React, { useState, useMemo } from 'react';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';

/**
 * Stochastic gene expression simulation using the Gillespie algorithm.
 * Simulates mRNA production/degradation and protein production/degradation
 * with optional repressor binding.
 */
export default function GeneExpressionNoise() {
  const [kmRNA, setKmRNA] = useState(10.0);
  const [gmRNA, setGmRNA] = useState(1.0);
  const [kpro, setKpro] = useState(1.0);
  const [gpro, setGpro] = useState(0.1);
  const [nRepressor, setNRepressor] = useState(0);
  const [seed, setSeed] = useState(0);

  const { tVals, mRNATrace, proteinTrace, repTrace, stats } = useMemo(() => {
    // Seeded pseudo-random number generator (simple LCG)
    let rngState = (seed + 1) * 2654435761;
    function random() {
      rngState = (rngState * 1664525 + 1013904223) & 0xffffffff;
      return (rngState >>> 0) / 4294967296;
    }

    const tFinal = 500;
    const nRecord = 500;
    const ra = 100;
    const kdiss = 1;
    const rd = ra * kdiss;
    const grep = 0.05;
    const krep = grep * nRepressor;

    const tVals: number[] = [];
    const mRNATrace: number[] = [];
    const proteinTrace: number[] = [];
    const repTrace: number[] = [];

    for (let i = 0; i < nRecord; i++) {
      tVals.push((i / (nRecord - 1)) * tFinal);
      mRNATrace.push(0);
      proteinTrace.push(0);
      repTrace.push(0);
    }

    const mRNAinit = Math.floor(kmRNA / (1.0 + nRepressor / kdiss) / gmRNA);
    const proInit = Math.floor(kpro * mRNAinit / gpro);

    let mRNAnow = mRNAinit;
    let proNow = proInit;
    let NRnow = nRepressor;
    let unoccupy = 1.0;
    let timeNow = 0.0;
    let itime = 0;

    const maxIter = 2000000;
    let iter = 0;

    while (timeNow < tFinal && iter < maxIter) {
      iter++;
      const bind = unoccupy * NRnow * ra;
      const unbind = (1 - unoccupy) * rd;
      const mprod = kmRNA * unoccupy;
      const mdeg = mRNAnow * gmRNA;
      const pprod = mRNAnow * kpro;
      const pdeg = proNow * gpro;
      const repdeg = grep * NRnow;

      const rsum = krep + repdeg + mprod + mdeg + pprod + pdeg + bind + unbind;
      if (rsum <= 0) break;

      const a1 = random();
      const tau = -Math.log(a1 + 1e-30) / rsum;

      // Record state during this interval
      while (itime < nRecord && tVals[itime] >= timeNow && tVals[itime] < timeNow + tau) {
        mRNATrace[itime] = mRNAnow;
        proteinTrace[itime] = proNow;
        repTrace[itime] = NRnow;
        itime++;
      }

      timeNow += tau;

      // Determine which reaction fires
      const a2 = random();
      let cumSum = krep / rsum;
      if (a2 < cumSum) {
        NRnow++;
      } else {
        cumSum += repdeg / rsum;
        if (a2 < cumSum) {
          NRnow = Math.max(0, NRnow - 1);
        } else {
          cumSum += mprod / rsum;
          if (a2 < cumSum) {
            mRNAnow++;
          } else {
            cumSum += mdeg / rsum;
            if (a2 < cumSum) {
              mRNAnow = Math.max(0, mRNAnow - 1);
            } else {
              cumSum += pprod / rsum;
              if (a2 < cumSum) {
                proNow++;
              } else {
                cumSum += pdeg / rsum;
                if (a2 < cumSum) {
                  proNow = Math.max(0, proNow - 1);
                } else {
                  cumSum += bind / rsum;
                  if (a2 < cumSum) {
                    unoccupy = 0;
                    NRnow = Math.max(0, NRnow - 1);
                  } else {
                    unoccupy = 1;
                    NRnow++;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Fill remaining time points
    while (itime < nRecord) {
      mRNATrace[itime] = mRNAnow;
      proteinTrace[itime] = proNow;
      repTrace[itime] = NRnow;
      itime++;
    }

    // Compute statistics (use second half for steady-state stats)
    const halfN = Math.floor(nRecord / 2);
    const proSS = proteinTrace.slice(halfN);
    const mrnaSS = mRNATrace.slice(halfN);
    const proMean = proSS.reduce((a, b) => a + b, 0) / proSS.length;
    const mrnaMean = mrnaSS.reduce((a, b) => a + b, 0) / mrnaSS.length;
    const proVar = proSS.reduce((a, b) => a + (b - proMean) ** 2, 0) / proSS.length;
    const proStd = Math.sqrt(proVar);
    const proNoise = proMean > 0 ? proStd / proMean : 0;

    return {
      tVals,
      mRNATrace,
      proteinTrace,
      repTrace,
      stats: {
        mrnaMean: mrnaMean.toFixed(2),
        proMean: proMean.toFixed(2),
        proVar: proVar.toFixed(2),
        proStd: proStd.toFixed(2),
        proNoise: proNoise.toFixed(4),
      },
    };
  }, [kmRNA, gmRNA, kpro, gpro, nRepressor, seed]);

  const commonLayout = {
    margin: { t: 40, b: 50, l: 60, r: 20 },
    xaxis: {
      title: { text: 'Time' },
    },
    yaxis: {},
  };

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">Stochastic Gene Expression (Gillespie Algorithm)</h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">mRNA production rate k_mRNA: {kmRNA.toFixed(1)}</label>
          <Slider value={[kmRNA]} onValueChange={([v]) => setKmRNA(v)} min={1} max={50} step={1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">mRNA degradation rate g_mRNA: {gmRNA.toFixed(1)}</label>
          <Slider value={[gmRNA]} onValueChange={([v]) => setGmRNA(v)} min={0.1} max={5} step={0.1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Protein production rate k_pro: {kpro.toFixed(1)}</label>
          <Slider value={[kpro]} onValueChange={([v]) => setKpro(v)} min={0.1} max={5} step={0.1} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Protein degradation rate g_pro: {gpro.toFixed(2)}</label>
          <Slider value={[gpro]} onValueChange={([v]) => setGpro(v)} min={0.01} max={1} step={0.01} />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Number of Repressors: {nRepressor}</label>
          <Slider value={[nRepressor]} onValueChange={([v]) => setNRepressor(v)} min={0} max={50} step={1} />
        </div>
        <div>
          <button
            onClick={() => setSeed((s) => s + 1)}
            className="mt-4 px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm"
          >
            Re-run simulation
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CanvasChart
          data={[
            {
              x: tVals, y: mRNATrace, type: 'scatter', mode: 'lines',
              line: { color: '#3b82f6', width: 1.5 }, name: 'mRNA',
            },
          ] as any}
          layout={{
            ...commonLayout,
            height: 300,
            title: { text: 'mRNA over time' },
            yaxis: { ...commonLayout.yaxis, title: { text: 'mRNA count' } },
          }}
          style={{ width: '100%' }}
        />
        <CanvasChart
          data={[
            {
              x: tVals, y: proteinTrace, type: 'scatter', mode: 'lines',
              line: { color: '#ef4444', width: 1.5 }, name: 'Protein',
            },
          ] as any}
          layout={{
            ...commonLayout,
            height: 300,
            title: { text: 'Protein over time' },
            yaxis: { ...commonLayout.yaxis, title: { text: 'Protein count' } },
          }}
          style={{ width: '100%' }}
        />
      </div>

      {nRepressor > 0 && (
        <CanvasChart
          data={[
            {
              x: tVals, y: repTrace, type: 'scatter', mode: 'lines',
              line: { color: '#22c55e', width: 1.5 }, name: 'Free Repressor',
            },
          ] as any}
          layout={{
            ...commonLayout,
            height: 250,
            title: { text: 'Free repressor over time' },
            yaxis: { ...commonLayout.yaxis, title: { text: 'Repressor count' } },
          }}
          style={{ width: '100%' }}
        />
      )}

      <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">mRNA avg</div>
          <div className="text-[var(--text-strong)] font-mono">{stats.mrnaMean}</div>
        </div>
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">Protein avg</div>
          <div className="text-[var(--text-strong)] font-mono">{stats.proMean}</div>
        </div>
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">Protein var</div>
          <div className="text-[var(--text-strong)] font-mono">{stats.proVar}</div>
        </div>
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">Protein std</div>
          <div className="text-[var(--text-strong)] font-mono">{stats.proStd}</div>
        </div>
        <div className="bg-[var(--surface-1)] border border-[var(--border-strong)] rounded p-3">
          <div className="text-[var(--text-soft)]">Noise (CV)</div>
          <div className="text-[var(--text-strong)] font-mono">{stats.proNoise}</div>
        </div>
      </div>
    </div>
  );
}
