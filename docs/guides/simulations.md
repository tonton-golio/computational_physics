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

## Do / Don't

### Do

- keep simulation ids stable and unique across all registries
- use lazy imports through manifest groups
- run integrity checks on every placeholder/registry change
- add descriptions for PDF export

### Don't

- hardcode simulation component imports in markdown renderer
- bypass manifest by dynamic lookup hacks
- ignore fallback behavior when testing unknown ids
