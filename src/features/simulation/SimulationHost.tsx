"use client";

import { useEffect, useRef, useState } from "react";
import type { SimulationComponent } from "@/shared/types/simulation";
import { prefetchSimulationDefinition, resolveSimulationDefinition } from "./simulation-manifest";

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
    <div className="flex h-80 w-full items-center justify-center rounded-xl border border-[var(--border-strong)] bg-[var(--surface-2)]">
      <div className="font-mono text-sm text-[var(--text-soft)]">Loading simulation...</div>
    </div>
  );
}

function SimulationError({ id }: { id: string }) {
  return (
    <div className="flex h-64 w-full flex-col items-center justify-center rounded-xl border border-[var(--danger-border)] bg-[var(--danger-surface)] text-[var(--text-soft)]">
      <span className="mb-2 text-[var(--accent-strong)]">Unable to render simulation</span>
      <span className="text-sm">[[simulation {id}]]</span>
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
    const timeoutId = window.setTimeout(warm, 120);
    return () => window.clearTimeout(timeoutId);
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

  return (
    <div
      ref={containerRef}
      className="w-full"
      onMouseEnter={() => setHasUserIntent(true)}
      onFocus={() => setHasUserIntent(true)}
      onClick={() => setHasUserIntent(true)}
      onTouchStart={() => setHasUserIntent(true)}
    >
      {!isVisible && !hasUserIntent ? (
        <SimulationLoading />
      ) : Simulation ? (
        <Simulation id={id} />
      ) : Graph ? (
        <Graph type={id} />
      ) : hasError ? (
        <SimulationError id={id} />
      ) : (
        <SimulationLoading />
      )}
    </div>
  );
}
