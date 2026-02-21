# Simulations Guide

## Purpose

Explain how simulation placeholders map to actual React components and how to add new simulations safely.

## Runtime Pieces

| File | Role |
|---|---|
| `src/components/content/MarkdownContent.tsx` | Placeholder extraction and React rendering |
| `src/features/simulation/SimulationHost.tsx` | Viewport-aware lazy loading with fallback |
| `src/features/simulation/simulation-manifest.ts` | Manifest-based id resolution across registries |
| `src/components/visualization/*Simulations.tsx` | Per-topic simulation registries (11 total) |

## Simulation Registries (11 groups)

| Registry File | Topic | Components |
|---|---|---|
| `AppliedStatsSimulations.tsx` | `applied-statistics` | 6 |
| `ComplexPhysicsSimulations.tsx` | `complex-physics` | 15 |
| `ContinuumMechanicsSimulations.tsx` | `continuum-mechanics` | 6 |
| `InverseProblemsSimulations.tsx` | `inverse-problems` | 9 |
| `QuantumOpticsSimulations.tsx` | `quantum-optics` | 5 |
| `DynamicalModelsSimulations.tsx` | `dynamical-models` | 9 |
| `AdvancedDeepLearningSimulations.tsx` | `advanced-deep-learning` | 11 |
| `AppliedMachineLearningSimulations.tsx` | `applied-machine-learning` | 8 |
| `ScientificComputingSimulations.tsx` | `scientific-computing` | 5 (includes nonlinear-equations) |
| `OnlineReinforcementSimulations.tsx` | `online-reinforcement-learning` | 18 |
| `EigenvalueSimulations.tsx` | `scientific-computing` | 10 (uses `next/dynamic` for SSR-unsafe Three.js) |

Each registry exports:
- `<TOPIC>_SIMULATIONS`: `Record<string, ComponentType<SimulationComponentProps>>` — the id-to-component map
- `<TOPIC>_DESCRIPTIONS`: `Record<string, string>` — human-readable descriptions (used in PDF export)
- `<TOPIC>_FIGURES` (some topics): `Record<string, { src, caption }>` — figure metadata

## Placeholder Contract

Lesson markdown uses:

```md
[[simulation simulation-id]]
```

This `simulation-id` must exist as a key in one topic registry object.

## Add A New Simulation

1. Create component in the appropriate `src/components/visualization/<topic>/` directory.
2. Register id in the topic `*Simulations.tsx` export map: `'simulation-id': SimulationComponent`
3. Add a description to the `*_DESCRIPTIONS` export.
4. Ensure topic registry module is included in `registryGroups` in `simulation-manifest.ts`.
5. Reference placeholder in markdown lesson: `[[simulation simulation-id]]`
6. Run: `npm run check:simulations` and `npm test`

## Charting Components

Simulations use pure-canvas chart components (not Plotly):

- `src/components/ui/canvas-chart.tsx` — 2D charts (scatter, line, bar, histogram)
- `src/components/ui/canvas-heatmap.tsx` — heatmaps with colorbar
- Both import shared theme from `src/lib/canvas-theme.ts`

Chart types: `ChartTrace` (with `type: 'scatter' | 'histogram' | 'bar'`), `ChartLayout`, `HeatmapData`, `HeatmapLayout`.

## Web Workers

Two simulations use Web Workers for heavy computation:

- `src/workers/simulation/lotka-volterra.worker.ts` — Euler-method ODE solver
- `src/workers/simulation/mdp-simulation.worker.ts` — MDP cost simulation

Client wrapper: `src/features/simulation/simulation-worker.client.ts` (LRU cache, 120 entries).

Shared param/result types live in `src/shared/types/simulation.ts`.

## Fallback And Failure Behavior

- If id resolves, host renders the loaded simulation component.
- If id does not resolve, host attempts `InteractiveGraph` fallback.
- If both paths fail, host renders a visible simulation error block.
- Manifest catches per-registry import failures so a single broken registry is isolated.

## Performance Behavior

- Simulations are not eagerly loaded for all placeholders.
- `SimulationHost` uses two intersection observers:
  - 1200px margin: triggers prefetch (background import)
  - 200px margin: triggers load (renders component)
- User intent (mouseenter, focus, click, touchstart) also forces load.
- First render timing emits a `simulation-first-render` browser event for observability.
- Online-reinforcement simulations have a fast-path set (`ONLINE_REINFORCEMENT_IDS`) to avoid scanning all registries.

## Fullscreen Layout

Simulations use the browser Fullscreen API. `SimulationHost` wraps content in a `SimulationFullscreenProvider` so child components can detect fullscreen mode via `useSimulationFullscreen()`.

Three component categories participate in fullscreen layout:

| Component | Role | Fullscreen behavior |
|---|---|---|
| `<SimulationMain>` | Primary visualization wrapper | Fills the viewport (CSS class `.sim-fs-main`) |
| `<CanvasChart>` | Auxiliary chart | Overlays bottom-right, stacked vertically |
| `<SimulationPanel>` / `<Card>` | Controls | Overlays bottom-left with glass background |

### `SimulationMain` — primary visualization wrapper

**File:** `src/components/ui/simulation-main.tsx`

Wrap the primary visualization in `<SimulationMain>` when the simulation also includes `CanvasChart` components (hybrid simulations). If the simulation has no CanvasCharts, the CSS fallback handles fullscreen automatically.

```tsx
import { SimulationMain } from '@/components/ui/simulation-main';
```

| Prop | Type | Default | Description |
|---|---|---|---|
| `children` | `ReactNode` | — | The primary visualization (Canvas, p5.js, etc.) |
| `className` | `string` | — | Normal-mode class names (dropped in fullscreen) |
| `style` | `CSSProperties` | — | Normal-mode inline styles like `height` (dropped in fullscreen) |
| `scaleMode` | `"fill" \| "contain"` | `"fill"` | How the canvas scales in fullscreen |

**Scale modes:**

- `"fill"` — canvas fills the entire viewport area. Use for **Three.js** / React-Three-Fiber scenes that resize to their container automatically.
- `"contain"` — canvas is constrained to fit without overflow. Use for **square / fixed-aspect-ratio** canvases (p5.js, custom 2D GridCanvas, etc.).

### Auxiliary charts (`CanvasChart`)

`CanvasChart` automatically sets `data-fs-role="chart"` in fullscreen. No wrapper needed. In fullscreen, charts are positioned as individual stacked overlays in the bottom-right corner (up to 2×2 grid for 4 charts).

### Controls (`SimulationPanel` / `Card`)

These automatically set `data-fs-role="controls"` in fullscreen. No additional markup needed. Controls overlay bottom-left with a glass background.

### Examples

**Three.js + CanvasChart:**

```tsx
<SimulationMain className="w-full rounded-lg overflow-hidden" style={{ height: 400 }}>
  <Canvas camera={...}>
    <MyScene />
    <OrbitControls />
  </Canvas>
</SimulationMain>

<CanvasChart data={chartData} layout={chartLayout} style={{ height: 300 }} />
```

**p5.js canvas + CanvasChart in a grid:**

```tsx
<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
  <SimulationMain scaleMode="contain" className="rounded-lg overflow-hidden" style={{ background: ... }}>
    <MyP5Canvas />
  </SimulationMain>
  <CanvasChart data={chartData} layout={chartLayout} style={{ height: 360 }} />
</div>
```

**Multiple mains (side by side) + aux charts:**

```tsx
<SimulationMain style={{ height: 400 }}>
  <Canvas3DScene1 />
</SimulationMain>
<SimulationMain style={{ height: 400 }}>
  <Canvas3DScene2 />
</SimulationMain>
<CanvasChart data={chartData} layout={chartLayout} style={{ height: 300 }} />
```

**p5.js heatmap + 4 aux charts:**

```tsx
<SimulationMain scaleMode="contain" className="rounded-lg overflow-hidden" style={{ background: ... }}>
  <HeatmapCanvas />
</SimulationMain>

<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
  <CanvasChart ... />
  <CanvasChart ... />
</div>
<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
  <CanvasChart ... />
  <CanvasChart ... />
</div>
```

**SVG tree/diagram (no charts):**

```tsx
<SimulationMain scaleMode="contain" className="overflow-x-auto">
  <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ maxHeight: 500 }}>
    {/* SVG elements */}
  </svg>
</SimulationMain>
```

**Canvas heatmap + trail overlay + optional chart:**

```tsx
<SimulationMain scaleMode="contain" className="relative">
  <CanvasHeatmap data={heatData} layout={heatLayout} style={{ width: '100%', height: 420 }} />
  {trail.length > 0 && <TrailOverlay trail={trail} />}
</SimulationMain>
{showChart && <CanvasChart data={chartData} layout={chartLayout} style={{ height: 420 }} />}
```

**2D canvas with ResizeObserver + chart:**

```tsx
<SimulationMain scaleMode="contain" className="w-full">
  <div ref={containerRef} style={{ width: '100%' }}>
    <canvas ref={canvasRef} style={{ display: 'block' }} />
  </div>
</SimulationMain>
<CanvasChart data={chartData} layout={chartLayout} style={{ height: 300 }} />
```

### How it works

1. **`SimulationMain`** reads the fullscreen context. In normal mode it renders with the author's `className` and `style`. In fullscreen mode it drops those and applies the `.sim-fs-main` CSS class, which absolutely positions the element to fill the viewport.
2. **CSS** in `globals.css` handles all layout: `.sim-fs-main` fills the screen, `[data-fs-role="chart"]` overlays bottom-right, `[data-fs-role="controls"]` overlays bottom-left. Grid wrappers containing role elements get `display: contents` to become transparent.
3. **Minimal JS** in `SimulationHost` (deferred one frame via `requestAnimationFrame`) assigns `data-fs-chart-index` to each chart for CSS stacking, and splits multiple mains side by side.
4. **Fallback for canvas/video:** If no `SimulationMain` is used, the CSS rule `.sim-fs-content > * > *:has(canvas)` expands canvas-containing children to fill the viewport. Video elements have an equivalent rule.
5. **Fallback for SVG:** Inline SVGs with a `viewBox` attribute get the same treatment via `.sim-fs-content > * > *:has(> svg[viewBox])`. The SVG is scaled to fill the container.
6. **Chart-only promotion:** When no `SimulationMain` and no plain canvas/video exists, `SimulationHost` promotes all charts to act as mains — filling the viewport side by side.

### Simulation categories and fullscreen behavior

| Category | Description | Fullscreen handling |
|---|---|---|
| **Hybrid** (canvas + chart) | Primary viz (canvas, Three.js, SVG) plus `CanvasChart` auxiliaries | Wrap primary viz in `<SimulationMain>` |
| **Chart-only** | Only `CanvasChart` components, no raw canvas/SVG | Charts auto-promoted to fill viewport |
| **Canvas-only** | Single canvas (p5.js, 2D, etc.), no charts | CSS fallback (`:has(canvas)`) fills viewport |
| **SVG-only** | Inline SVG visualization, no canvas or charts | CSS fallback (`:has(> svg[viewBox])`) fills viewport |
| **Three.js + charts** | R3F `<Canvas>` plus `CanvasChart` auxiliaries | Wrap `<Canvas>` in `<SimulationMain>` with `scaleMode="fill"` |

**When do you need `<SimulationMain>`?** Only for hybrid simulations that combine a primary visualization with auxiliary `CanvasChart` components. Non-hybrid simulations (chart-only, canvas-only, SVG-only) work automatically via CSS fallbacks.

### Currently using `SimulationMain`

| Simulation | scaleMode | Visualization | Aux charts |
|---|---|---|---|
| `IsingModel` | `fill` | Three.js 3D spin grid | 2 |
| `ScaleFreeNetwork` | `fill` | Three.js 3D network | 2 |
| `SandpileModel` | `fill` | Three.js 3D height field | 2 |
| `PercolationSim` | `contain` | 2D GridCanvas (square) | 1 |
| `FractalDimension` | `contain` | p5.js FractalCanvas | 1 |
| `BetheLatticePercolation` | `contain` | 2D TreeCanvas | 1 |
| `RandomWalkFirstReturn` | `contain` | p5.js WalkCanvas (1:1) | 1 |
| `BakSneppen` | `contain` | p5.js EvolutionCanvas heatmap | 4 |
| `AdversarialAttack` | `contain` | 2D pixel-art canvas | 2 |
| `FilterEvolution` | `contain` | 2D filter grid canvas | 1 |
| `ReceptiveFieldGrowth` | `contain` | 2D receptive field grid | 1 |
| `StokesFlowDemo` | `contain` | 2D streamline canvas | 1 |
| `LossLandscapeExplorer` | `contain` | CanvasHeatmap + trail overlay | 1 (optional) |
| `BacterialLineageTree` | `contain` | SVG tree diagram | 0 |

## 3D Simulations (Three.js / React Three Fiber)

3D simulations use `@react-three/fiber` (R3F) with helpers from `@react-three/drei`. The key libraries:

| Package | Version | Role |
|---|---|---|
| `three` | 0.182 | Core 3D engine |
| `@react-three/fiber` | 9.x | React renderer for Three.js |
| `@react-three/drei` | 10.x | Helpers: `OrbitControls`, `Html`, `Line`, etc. |
| `@react-three/postprocessing` | 3.x | Bloom and other post-processing effects |

### Component structure

Split the 3D scene into a separate `Scene` component that contains all R3F hooks and objects. The parent component owns state, sliders, and the `<Canvas>` wrapper.

```tsx
'use client';

import { useState, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '@/lib/use-theme';

// Scene component — all R3F hooks must live inside <Canvas>
function Scene({ data, isDark }: { data: MyData; isDark: boolean }) {
  return (
    <>
      <ambientLight intensity={isDark ? 0.3 : 0.6} />
      <directionalLight position={[5, 8, 5]} intensity={isDark ? 0.5 : 0.8} />

      {/* 3D objects here */}

      <OrbitControls enableDamping dampingFactor={0.08} />
    </>
  );
}

// Main component — owns state, renders Canvas + controls
export default function MySimulation() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  // ... state, sliders, etc.

  return (
    <SimulationPanel>
      {/* Sliders / controls */}

      <div style={{ height: 420, background: isDark ? '#0a0a14' : '#f8fafc' }}>
        <Canvas
          camera={{ position: [3, 2.5, 3], fov: 50, near: 0.1, far: 100 }}
          dpr={[1, 2]}
          style={{ width: '100%', height: '100%' }}
        >
          <color attach="background" args={[isDark ? '#0a0a14' : '#f8fafc']} />
          <Scene data={data} isDark={isDark} />
        </Canvas>
      </div>
    </SimulationPanel>
  );
}
```

### Camera and controls

Use `OrbitControls` from drei for interactive drag-to-rotate. Do **not** add manual view-angle sliders — OrbitControls replaces them.

```tsx
<OrbitControls
  enableDamping           // smooth deceleration after drag
  dampingFactor={0.08}
  minDistance={2}          // zoom limits
  maxDistance={12}
  makeDefault             // set as default camera control
/>
```

Optional props: `autoRotate`, `autoRotateSpeed`, `enablePan`, `enableZoom`, `enableRotate`.

### Dark mode and lighting

Use `useTheme()` from `@/lib/use-theme` (returns `'dark' | 'light'`). Adjust:

- **Background**: set both the container `background` and `<color attach="background">` inside Canvas
- **Ambient light**: lower intensity in dark mode (~0.3 vs ~0.6)
- **Directional lights**: adjust intensity and optionally add tinted fill lights

```tsx
const theme = useTheme();
const isDark = theme === 'dark';
```

### Post-processing (Bloom)

Apply bloom in dark mode only for a glowing effect. Wrap in `<EffectComposer>`:

```tsx
import { EffectComposer, Bloom } from '@react-three/postprocessing';

{isDark && (
  <EffectComposer>
    <Bloom
      luminanceThreshold={0.2}
      luminanceSmoothing={0.9}
      intensity={1.4}
      mipmapBlur
    />
  </EffectComposer>
)}
```

Materials that should glow need `toneMapped={false}` and an `emissive` + `emissiveIntensity`.

### Instanced meshes (many identical objects)

For grids, particle systems, or networks with many objects (100+), use `<instancedMesh>` for performance:

```tsx
const dummy = new THREE.Object3D();

function MyGrid({ data }: { data: number[][] }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = data.length * data[0].length;

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    for (let i = 0; i < count; i++) {
      dummy.position.set(x, y, z);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }, [data]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <boxGeometry args={[0.9, 0.5, 0.9]} />
      <meshStandardMaterial vertexColors />
    </instancedMesh>
  );
}
```

### Labels in 3D

Use `<Html>` from drei for text labels that stay screen-aligned and readable at any angle:

```tsx
import { Html } from '@react-three/drei';

<Html position={[x, y, z]} center style={{ pointerEvents: 'none' }}>
  <span style={{ color: '#ef4444', fontSize: 13, fontWeight: 'bold', fontFamily: 'monospace' }}>
    Label
  </span>
</Html>
```

### Lines and arrows

Use `<Line>` from drei for lines with controllable width (native WebGL lines are always 1px):

```tsx
import { Line } from '@react-three/drei';

<Line
  points={[[0,0,0], [1,1,1]]}
  color="#8b5cf6"
  lineWidth={2.5}
  dashed              // optional
  dashSize={0.12}
  gapSize={0.06}
/>
```

For arrows, combine a `<Line>` shaft with a `<coneGeometry>` arrowhead, aligned via quaternion:

```tsx
const dir = new THREE.Vector3().subVectors(toV, fromV).normalize();
const quaternion = new THREE.Quaternion().setFromUnitVectors(
  new THREE.Vector3(0, 1, 0), dir
);

<mesh position={conePosition} quaternion={quaternion}>
  <coneGeometry args={[radius, height, 8]} />
  <meshBasicMaterial color={color} />
</mesh>
```

### Animation (useFrame)

For continuous animation or physics updates, use the `useFrame` hook inside a Scene component:

```tsx
import { useFrame } from '@react-three/fiber';

useFrame((state, delta) => {
  groupRef.current.rotation.y += delta * 0.05;
});
```

### Fullscreen with SimulationMain

When combining a 3D canvas with `CanvasChart` components, wrap the canvas in `<SimulationMain>`:

```tsx
import { SimulationMain } from '@/components/ui/simulation-main';

<SimulationMain
  className="w-full rounded-lg overflow-hidden"
  style={{ height: 400, background: isDark ? '#0a0a0f' : '#f0f4ff' }}
>
  <Canvas ...>
    <Scene />
  </Canvas>
</SimulationMain>
```

Use `scaleMode="fill"` (default) for Three.js scenes that resize to their container.

### Existing 3D simulations (reference examples)

| Component | Pattern | Key technique |
|---|---|---|
| `LorenzAttractor` | Tube geometry + vertex colors | `TubeGeometry`, `CatmullRomCurve3`, velocity-based coloring |
| `IsingModel` | Instanced grid | `<instancedMesh>`, per-instance color via `InstancedBufferAttribute` |
| `SandpileModel` | Animated height field | `useFrame` animation loop, instanced cubes |
| `ScaleFreeNetwork` | Node-link 3D graph | `useFrame` for slow rotation, instanced spheres + line geometry |
| `GeometricProjection` | Vector arrows + plane | `<Line>` + cone arrowheads, `<Html>` labels, custom quad geometry |

## Do / Don't

### Do

- keep simulation ids stable and unique across all registries
- use lazy imports through manifest groups
- run integrity checks on every placeholder/registry change
- add descriptions for PDF export
- wrap the primary viz in `<SimulationMain>` for hybrid simulations
- use `scaleMode="contain"` for square or fixed-aspect-ratio canvases
- let `CanvasChart` and `SimulationPanel`/`Card` self-identify (no manual `data-fs-role`)
- use `OrbitControls` for 3D view rotation instead of manual azimuth/elevation sliders
- use `<instancedMesh>` for scenes with many identical objects (100+)
- use `useTheme()` and adjust lighting/background for dark mode
- keep R3F hooks (`useFrame`, `useThree`) inside components rendered within `<Canvas>`
- set `dpr={[1, 2]}` on Canvas for retina support without excessive GPU load

### Don't

- hardcode simulation component imports in markdown renderer
- bypass manifest by dynamic lookup hacks
- ignore fallback behavior when testing unknown ids
- manually set `data-fs-role="main"` on a div (use `<SimulationMain>` instead)
- set `data-fs-role="chart"` or `data-fs-role="controls"` manually (components do this)
- use React hooks (useFrame, useThree) outside a `<Canvas>` — they only work inside the R3F context
- add view-angle sliders when OrbitControls provides drag-to-rotate
- create individual `<mesh>` elements for hundreds of objects — use instancing instead
- forget `toneMapped={false}` on materials that should participate in Bloom
