"use client";

import { createContext, useContext } from "react";
import { useSimulationFullscreen } from "@/lib/simulation-fullscreen-context";
import { cn } from "@/lib/utils";

/* ── Context ─────────────────────────────────────────────────────────── */

/**
 * Discriminated context:
 * - `false`  — outside both Main and Aux (charts become floating overlays)
 * - `'main'` — inside SimulationMain (charts fill the viewport in fullscreen)
 * - `'aux'`  — inside SimulationAux (charts keep natural size, no overlay)
 */
export const SimulationMainContext = createContext<false | 'main' | 'aux'>(false);

/**
 * Returns the enclosing slot: `'main'`, `'aux'`, or `false`.
 *
 * Truthy when the calling component is rendered inside a
 * `<SimulationMain>` or `<SimulationAux>` wrapper.
 * Used by CanvasChart / CanvasHeatmap to suppress their
 * `data-fs-role="chart"` overlay behavior.
 */
export function useIsInsideMain() {
  return useContext(SimulationMainContext);
}

/* ── Component ───────────────────────────────────────────────────────── */

interface SimulationMainProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  /**
   * How the primary canvas scales in fullscreen.
   *
   * - `"fill"` (default) — canvas fills the entire viewport area.
   *   Use for Three.js / React-Three-Fiber scenes that resize automatically.
   *
   * - `"contain"` — canvas is constrained to fit without overflow.
   *   Use for square / fixed-aspect-ratio canvases (p5.js, custom 2D).
   */
  scaleMode?: "fill" | "contain";
}

/**
 * Wrapper for the primary visualization in a simulation.
 *
 * In normal mode the wrapper renders with the author's className and style.
 * In fullscreen mode it switches to the `.sim-fs-main` CSS class, which
 * fills the viewport and centers the canvas.
 *
 * Children (CanvasChart, CanvasHeatmap) detect this context via
 * `useIsInsideMain()` and suppress their chart-overlay behavior.
 *
 * @example
 * ```tsx
 * <SimulationMain className="w-full rounded-lg overflow-hidden" style={{ height: 400 }}>
 *   <Canvas camera={...}><MyScene /></Canvas>
 * </SimulationMain>
 * ```
 */
export function SimulationMain({
  children,
  className,
  style,
  scaleMode = "fill",
}: SimulationMainProps) {
  const isFullscreen = useSimulationFullscreen();

  return (
    <SimulationMainContext.Provider value="main">
      <div
        data-fs-role="main"
        data-sim-slot="main"
        data-fs-scale={scaleMode}
        className={cn(
          !isFullscreen && className,
          isFullscreen && "sim-fs-main",
        )}
        style={isFullscreen ? undefined : style}
      >
        {children}
      </div>
    </SimulationMainContext.Provider>
  );
}
