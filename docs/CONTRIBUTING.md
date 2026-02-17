# Contributing Guide

## Purpose

Define a reliable contribution workflow for humans and autonomous agents.

## Scope

This guide applies to code and docs work inside `computational_physics/`.

## Required Pre-Change Checks

- Confirm task intent and constraints from:
  - `.claw/claw/AGENTS.md`
  - `../../AGENTS.md`
- Identify impacted runtime area:
  - content pipeline
  - simulation runtime
  - API contract
  - operations/runbook docs

## Development Workflow

1. Make focused changes in smallest meaningful unit.
2. Run the minimum relevant checks:
   - `npm run lint`
   - `npm run check:simulations` (required for content/simulation changes)
   - `npm test`
   - `npm run test:e2e` for route/rendering regressions
3. Update docs in `docs/` when behavior, contract, or workflow changed.
4. Validate no broken links or stale command names in docs.

## Definition Of Done

- Builds and checks pass for impacted surfaces.
- API/error contract changes reflected in `docs/guides/api-contracts.md`.
- Content/simulation integration changes reflected in:
  - `docs/guides/content-authoring.md`
  - `docs/guides/simulations.md`
- Operational behavior changes reflected in runbooks under `docs/operations` or `docs/runbooks`.

## Commit Expectations

- Keep commits focused and reversible.
- Prefer descriptive messages explaining why the change exists.
- Avoid bundling refactors with behavior changes unless coupled by necessity.

## Do / Don’t

### Do

- keep contracts typed and explicit
- preserve existing error envelope shape
- favor additive docs changes when introducing new workflows

### Don’t

- silently change API response shape
- add undocumented new placeholder syntax
- merge behavior changes without corresponding docs updates
