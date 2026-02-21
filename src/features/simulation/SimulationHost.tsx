"use client";

import { useEffect, useRef, useState } from "react";
import type { SimulationComponent } from "@/shared/types/simulation";
import { prefetchSimulationDefinition, resolveSimulationDefinition } from "./simulation-manifest";
import { useFullscreen } from "@/lib/use-fullscreen";
import { FullscreenButton } from "@/components/ui/fullscreen-button";
import { SimulationFullscreenProvider } from "@/lib/simulation-fullscreen-context";
import { SIMULATION_DESCRIPTIONS } from "@/lib/simulation-descriptions";
import { InlineSuggestionBox } from "@/components/layout/InlineSuggestionBox";

type GraphComponent = React.ComponentType<{ type: string; params?: Record<string, number> }>;
type LoadResult = {
  simulation: SimulationComponent | null;
  graph: GraphComponent | null;
  hasError: boolean;
};

const renderCache = new Map<string, LoadResult>();
const inFlightRenderCache = new Map<string, Promise<LoadResult>>();
let graphComponentPromise: Promise<GraphComponent> | null = null;

function loadGraphComponent(): Promise<GraphComponent> {
  if (graphComponentPromise) return graphComponentPromise;
  graphComponentPromise = import("@/components/visualization/InteractiveGraph").then(
    (module) => module.InteractiveGraph as GraphComponent
  );
  return graphComponentPromise;
}

async function loadSimulation(id: string): Promise<LoadResult> {
  const cached = renderCache.get(id);
  if (cached) return cached;

  const inFlight = inFlightRenderCache.get(id);
  if (inFlight) return inFlight;

  const loadingPromise = (async (): Promise<LoadResult> => {
    const definition = await resolveSimulationDefinition(id);
    if (definition) {
      const component = await definition.load();
      if (component) return { simulation: component, graph: null, hasError: false };
    }

    const graphComponent = await loadGraphComponent();
    return { simulation: null, graph: graphComponent, hasError: false };
  })()
    .catch((): LoadResult => ({ simulation: null, graph: null, hasError: true }))
    .finally(() => {
      inFlightRenderCache.delete(id);
    });

  inFlightRenderCache.set(id, loadingPromise);
  const result = await loadingPromise;
  renderCache.set(id, result);
  return result;
}

function SimulationLoading() {
  return (
    <div className="relative h-80 w-full overflow-hidden rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)]">
      <div className="flex h-full w-full items-center justify-center">
        <div className="font-mono text-sm text-[var(--text-soft)] animate-pulse">Loading simulation…</div>
      </div>
      <div
        className="pointer-events-none absolute inset-0 -translate-x-full animate-[shimmer_1.8s_ease-in-out_infinite]"
        style={{
          background:
            "linear-gradient(90deg, transparent 0%, var(--surface-3) 50%, transparent 100%)",
        }}
      />
    </div>
  );
}

function SimulationError({ id }: { id: string }) {
  return (
    <div className="flex h-64 w-full flex-col items-center justify-center gap-2 rounded-xl border border-[var(--danger-border)] bg-[var(--danger-surface)] shadow-inner">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--danger-border)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span className="font-medium text-[var(--text-strong)]">Unable to render simulation</span>
      <span className="font-mono text-xs text-[var(--text-soft)]">[[simulation {id}]]</span>
    </div>
  );
}

export function SimulationHost({ id }: { id: string }) {
  const [Simulation, setSimulation] = useState<SimulationComponent | null>(null);
  const [Graph, setGraph] = useState<GraphComponent | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isNearViewport, setIsNearViewport] = useState(false);
  const [hasUserIntent, setHasUserIntent] = useState(false);
  const [hasError, setHasError] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const loadStartRef = useRef<number | null>(null);
  const measuredRef = useRef(false);

  useEffect(() => {
    loadStartRef.current = null;
    measuredRef.current = false;
  }, [id]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    if (typeof IntersectionObserver === "undefined") return;

    const observer = new IntersectionObserver(([entry]) => setIsVisible(entry.isIntersecting), {
      rootMargin: "200px 0px 200px 0px",
      threshold: 0.01,
    });
    const prefetchObserver = new IntersectionObserver(([entry]) => setIsNearViewport(entry.isIntersecting), {
      rootMargin: "1200px 0px 1200px 0px",
      threshold: 0.01,
    });

    observer.observe(element);
    prefetchObserver.observe(element);
    return () => {
      observer.disconnect();
      prefetchObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!isNearViewport) return;
    if (Simulation || Graph || hasError) return;
    const warm = () => {
      void prefetchSimulationDefinition(id);
    };
    if (typeof window !== "undefined" && "requestIdleCallback" in window) {
      const idleId = window.requestIdleCallback(warm, { timeout: 600 });
      return () => window.cancelIdleCallback(idleId);
    }
    const timeoutId = globalThis.setTimeout(warm, 120);
    return () => globalThis.clearTimeout(timeoutId);
  }, [id, isNearViewport, Simulation, Graph, hasError]);

  useEffect(() => {
    let active = true;
    if (!(isVisible || hasUserIntent) || Simulation || Graph) return;
    if (loadStartRef.current === null) {
      loadStartRef.current = performance.now();
    }

    void loadSimulation(id)
      .then((result) => {
        if (!active) return;
        if (result.hasError) {
          setHasError(true);
          return;
        }
        if (result.simulation) {
          setSimulation(() => result.simulation);
          return;
        }
        if (result.graph) {
          setGraph(() => result.graph);
          return;
        }
        setHasError(true);
      })
      .catch(() => {
        if (!active) return;
        setHasError(true);
      });

    return () => {
      active = false;
    };
  }, [id, isVisible, hasUserIntent, Simulation, Graph]);

  useEffect(() => {
    if (measuredRef.current) return;
    if (!Simulation && !Graph) return;
    const start = loadStartRef.current;
    if (start === null) return;
    const durationMs = Math.max(0, performance.now() - start);
    measuredRef.current = true;
    window.dispatchEvent(
      new CustomEvent("simulation-first-render", {
        detail: { id, durationMs },
      })
    );
  }, [id, Simulation, Graph]);

  const { isFullscreen, toggle: toggleFullscreen } = useFullscreen(containerRef);

  // Minimal fullscreen effect: index charts for CSS stacking, handle multi-main split,
  // and promote charts to mains when there is no primary visualization.
  // Layout is driven by CSS classes (.sim-fs-main, [data-fs-role] selectors).
  // Deferred by one frame so React has flushed data-fs-role attributes.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !isFullscreen) return;

    const cleanup: (() => void)[] = [];

    const rafId = requestAnimationFrame(() => {
      const fsContent = container.querySelector(".sim-fs-content");
      if (!fsContent) return;

      const mains = fsContent.querySelectorAll<HTMLElement>(".sim-fs-main");
      const charts = fsContent.querySelectorAll<HTMLElement>("[data-fs-role='chart']");

      // Fallback: chart-only simulations (no SimulationMain, no plain canvas/video).
      // Promote all charts to act as mains — fill the viewport side by side.
      const hasPlainCanvas = fsContent.querySelector(
        "canvas:not([data-fs-role] canvas), video:not([data-fs-role] video)",
      );
      if (mains.length === 0 && !hasPlainCanvas && charts.length > 0) {
        const n = charts.length;
        const slice = 100 / n;
        charts.forEach((c, i) => {
          c.classList.add("sim-fs-main");
          cleanup.push(() => c.classList.remove("sim-fs-main"));
          // Remove chart overlay positioning
          const prevRole = c.dataset.fsRole;
          c.dataset.fsRole = "main";
          cleanup.push(() => {
            if (prevRole) c.dataset.fsRole = prevRole;
          });
          // Side-by-side split for multiple charts
          if (n > 1) {
            const prevLeft = c.style.left;
            const prevWidth = c.style.width;
            c.style.left = `${i * slice}%`;
            c.style.width = `${slice}%`;
            cleanup.push(() => {
              c.style.left = prevLeft;
              c.style.width = prevWidth;
            });
          }
        });
        return;
      }

      // Multi-main side-by-side split
      if (mains.length > 1) {
        const slice = 100 / mains.length;
        mains.forEach((m, i) => {
          const prevLeft = m.style.left;
          const prevWidth = m.style.width;
          m.style.left = `${i * slice}%`;
          m.style.width = `${slice}%`;
          cleanup.push(() => {
            m.style.left = prevLeft;
            m.style.width = prevWidth;
          });
        });
      }

      // Index charts for CSS stacking (bottom-right grid layout)
      // Only index charts that remain as overlays (not promoted to mains)
      const overlayCharts = fsContent.querySelectorAll<HTMLElement>("[data-fs-role='chart']");
      overlayCharts.forEach((c, i) => {
        c.dataset.fsChartIndex = String(i);
        cleanup.push(() => delete c.dataset.fsChartIndex);
      });
    });

    return () => {
      cancelAnimationFrame(rafId);
      cleanup.forEach((fn) => fn());
    };
  }, [isFullscreen]);

  const simContent = !isVisible && !hasUserIntent ? (
    <SimulationLoading />
  ) : Simulation ? (
    <Simulation id={id} />
  ) : Graph ? (
    <Graph type={id} />
  ) : hasError ? (
    <SimulationError id={id} />
  ) : (
    <SimulationLoading />
  );

  return (
    <div
      ref={containerRef}
      className={
        isFullscreen
          ? "relative w-full h-full bg-[var(--surface-1)] overflow-hidden"
          : "relative w-full rounded-xl border border-[var(--border-strong)] bg-[var(--surface-1)] py-4 shadow-[inset_0_1px_2px_rgba(0,0,0,0.04)] overflow-hidden"
      }
      onMouseEnter={() => setHasUserIntent(true)}
      onFocus={() => setHasUserIntent(true)}
      onClick={() => setHasUserIntent(true)}
      onTouchStart={() => setHasUserIntent(true)}
    >
      {/* Accent stripe */}
      {!isFullscreen && (
        <div
          className="h-[3px] -mt-4 mb-4"
          style={{ background: "linear-gradient(90deg, var(--accent), var(--accent-strong))" }}
        />
      )}

      <div className={isFullscreen ? "sim-fs-content" : undefined}>
        <SimulationFullscreenProvider value={isFullscreen}>
          {simContent}
        </SimulationFullscreenProvider>
      </div>

      {/* Inline caption (normal mode only) */}
      {!isFullscreen && SIMULATION_DESCRIPTIONS[id] && (
        <div className="px-4 pt-2 pb-1 text-xs text-[var(--text-soft)]">
          {SIMULATION_DESCRIPTIONS[id]}
        </div>
      )}

      {/* Fullscreen title overlay */}
      {isFullscreen && SIMULATION_DESCRIPTIONS[id] && (
        <div className="sim-fs-title">
          {SIMULATION_DESCRIPTIONS[id]}
        </div>
      )}

      {isFullscreen && <InlineSuggestionBox />}

      <FullscreenButton
        isFullscreen={isFullscreen}
        onClick={toggleFullscreen}
        className={
          isFullscreen
            ? "absolute top-3 right-3 z-20"
            : "absolute top-5 right-3 z-10"
        }
      />
    </div>
  );
}
