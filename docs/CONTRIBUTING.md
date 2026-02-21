# Contributing Guide

## Scope

Code and docs work inside `computational_physics/`.

## Development Workflow

1. Make focused changes in smallest meaningful unit.
2. Run relevant checks:
   - `npm run lint`
   - `npm run check:simulations` (content/simulation changes)
   - `npm test`
   - `npm run test:e2e` (route/rendering regressions)
3. Update docs in `docs/` when behavior, contract, or workflow changed.

## Definition Of Done

- Builds and checks pass for impacted surfaces.
- API contract changes reflected in `docs/guides/api-contracts.md`.
- Content/simulation changes reflected in `docs/guides/content-authoring.md` or `docs/guides/simulations.md`.
- Keep commits focused, reversible, with descriptive messages.

## Do / Don't

### Do

- keep contracts typed and explicit
- preserve existing error envelope shape
- favor additive docs changes when introducing new workflows

### Don't

- silently change API response shape
- add undocumented new placeholder syntax
- merge behavior changes without corresponding docs updates
