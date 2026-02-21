"use client";

import { useSimulationFullscreen } from "@/lib/simulation-fullscreen-context";
import { cn } from "@/lib/utils";

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
 * Wrapper for the primary visualization in a hybrid simulation
 * (one that also includes CanvasChart auxiliaries).
 *
 * In normal mode the wrapper renders with the author's className and style.
 * In fullscreen mode it switches to the `.sim-fs-main` CSS class, which
 * fills the viewport and centers the canvas.
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
    <div
      data-fs-role="main"
      data-fs-scale={scaleMode}
      className={cn(
        !isFullscreen && className,
        isFullscreen && "sim-fs-main",
      )}
      style={isFullscreen ? undefined : style}
    >
      {children}
    </div>
  );
}
