# Simulations Guide

## Purpose

Explain how simulation placeholders map to actual React components and how to add new simulations safely.

## Runtime Pieces

- Placeholder extraction/render:
  - `src/components/content/MarkdownContent.tsx`
- Runtime host and lazy loading:
  - `src/features/simulation/SimulationHost.tsx`
- Manifest-based id resolution:
  - `src/features/simulation/simulation-manifest.ts`
- Topic registries:
  - `src/components/visualization/*Simulations.tsx`

## Placeholder Contract

Lesson markdown uses:

```md
[[simulation simulation-id]]
```

This `simulation-id` must exist as a key in one topic registry object.

## Add A New Simulation

1. Implement component in topic visualization module.
2. Register id in the topic `*Simulations.tsx` export map:
   - `'simulation-id': SimulationComponent`
3. Ensure topic registry module is included in `registryGroups` in `simulation-manifest.ts`.
4. Reference placeholder in markdown lesson.
5. Run:
   - `npm run check:simulations`
   - `npm test`

## Fallback And Failure Behavior

- If id resolves, host renders the loaded simulation component.
- If id does not resolve, host attempts `InteractiveGraph` fallback.
- If both paths fail, host renders a visible simulation error block.
- Manifest catches per-registry import failures so a single broken registry is isolated.

## Performance Behavior

- Simulations are not eagerly loaded for all placeholders.
- `SimulationHost` uses intersection observers to prefetch near viewport and load on intent/visibility.
- First render timing emits a `simulation-first-render` browser event for observability.

## Do / Don’t

### Do

- keep simulation ids stable and unique
- use lazy imports through manifest groups
- run integrity checks on every placeholder/registry change

### Don’t

- hardcode simulation component imports in markdown renderer
- bypass manifest by dynamic lookup hacks
- ignore fallback behavior when testing unknown ids
