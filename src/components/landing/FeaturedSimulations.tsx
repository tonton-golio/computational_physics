"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { SimulationHost } from "@/features/simulation/SimulationHost";
import { SIMULATION_DESCRIPTIONS } from "@/lib/simulation-descriptions";

interface FeaturedItem {
  id: string;
  title: string;
  href: string;
}

const FEATURED: FeaturedItem[] = [
  {
    id: "weighted-mean",
    title: "Weighted Mean: Precision Matters",
    href: "/topics/applied-statistics/introduction-concepts",
  },
  {
    id: "mandelbrot-fractal",
    title: "Mandelbrot Explorer",
    href: "/topics/complex-physics/fractals",
  },
  {
    id: "sandpile-model",
    title: "Abelian Sandpile Model",
    href: "/topics/complex-physics/selfOrganizedCriticality",
  },
];

const AUTOPLAY_MS = 8000;

export function FeaturedSimulations() {
  const [active, setActive] = useState(0);
  const [progress, setProgress] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const rafRef = useRef<number | null>(null);
  const startTimeRef = useRef(Date.now());

  const tick = useCallback(() => {
    const elapsed = Date.now() - startTimeRef.current;
    setProgress(Math.min(elapsed / AUTOPLAY_MS, 1));
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const startAutoplay = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    startTimeRef.current = Date.now();
    setProgress(0);
    rafRef.current = requestAnimationFrame(tick);
    intervalRef.current = setInterval(() => {
      setActive((prev) => (prev + 1) % FEATURED.length);
      startTimeRef.current = Date.now();
      setProgress(0);
    }, AUTOPLAY_MS);
  }, [tick]);

  useEffect(() => {
    startAutoplay();
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [startAutoplay]);

  const goto = (index: number) => {
    setActive(((index % FEATURED.length) + FEATURED.length) % FEATURED.length);
    startAutoplay();
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header row */}
      <div className="mb-3 flex items-center justify-between">
        <p className="font-mono text-xs uppercase tracking-[0.28em] text-[var(--accent)]">
          featured visualizations
        </p>
        <div className="flex items-center gap-2">
          {/* Progress bars */}
          <div className="mr-3 flex gap-1.5">
            {FEATURED.map((_, i) => (
              <button
                key={i}
                onClick={() => goto(i)}
                className="relative h-1.5 w-6 overflow-hidden rounded-full bg-[var(--text-soft)]/20"
                aria-label={`Go to visualization ${i + 1}`}
              >
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-[var(--accent)] transition-none"
                  style={{
                    width:
                      i === active
                        ? `${progress * 100}%`
                        : i < active || (active === 0 && i === FEATURED.length - 1 && progress > 0)
                          ? "100%"
                          : "0%",
                  }}
                />
              </button>
            ))}
          </div>
          <button
            onClick={() => goto(active - 1)}
            className="rounded-md border border-[var(--border-strong)] p-1.5 text-[var(--text-muted)] transition hover:text-[var(--text-strong)]"
            aria-label="Previous visualization"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
          </button>
          <button
            onClick={() => goto(active + 1)}
            className="rounded-md border border-[var(--border-strong)] p-1.5 text-[var(--text-muted)] transition hover:text-[var(--text-strong)]"
            aria-label="Next visualization"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
          </button>
        </div>
      </div>

      {/* Outer card */}
      <div className="relative flex-1 min-h-0 flex flex-col overflow-hidden rounded-lg border border-[var(--border-strong)]">
        {/* Simulation area — stacked slides */}
        <div className="relative flex-1 min-h-0">
          {FEATURED.map((item, i) => (
            <div
              key={item.id}
              className={`absolute inset-0 transition-opacity duration-500 ${
                i === active
                  ? "opacity-100 z-[1]"
                  : "opacity-0 z-0 pointer-events-none"
              }`}
            >
              <SimulationHost id={item.id} variant="embedded" />
            </div>
          ))}
        </div>

        {/* Info bar */}
        <div className="relative z-[2] flex items-center gap-4 border-t border-[var(--border-strong)] bg-[var(--surface-1)]/60 backdrop-blur-sm px-4 py-2.5">
          <p className="font-mono text-sm text-[var(--text-strong)] shrink-0">
            {FEATURED[active].title}
          </p>
          <p className="flex-1 min-w-0 truncate text-xs text-[var(--text-soft)]">
            {SIMULATION_DESCRIPTIONS[FEATURED[active].id]}
          </p>
          <Link
            href={FEATURED[active].href}
            className="shrink-0 rounded-md border border-[var(--accent)] px-3 py-1.5 font-mono text-xs text-[var(--accent)] transition hover:bg-[var(--accent)] hover:text-black"
          >
            explore
          </Link>
        </div>
      </div>
    </div>
  );
}
