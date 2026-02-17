# Testing Standards

## Purpose

Define minimum test expectations for reliable autonomous changes.

## Test Stack

- Unit/integration: Vitest (`npm test`)
- End-to-end smoke: Playwright (`npm run test:e2e`)
- Structural integrity:
  - simulation placeholder integrity (`npm run check:simulations`)
  - performance budget guard (`npm run perf:budget`)

## Test Layout

- Unit/domain tests live with source:
  - `src/domain/**/*.test.ts`
  - `src/features/**/*.test.ts`
  - `src/shared/**/*.test.ts`
- E2E tests live in:
  - `tests/e2e/*.spec.ts`

## Required By Change Type

### Content or simulation placeholder changes

- `npm run check:simulations`
- `npm test`

### API route or contract changes

- `npm test`
- add or update route-level tests where practical
- verify `/api/health`, `/api/ready`, and changed routes manually or via e2e

### Navigation, page rendering, or UI changes

- `npm test`
- `npm run test:e2e`

## Test Design Rules

- Prefer deterministic tests with explicit fixtures.
- Assert contract shape, not incidental implementation details.
- Keep tests focused on behavior at module boundary.
- Add regression tests when fixing bugs.

## Reliability Checkpoints

- Ensure content index loading still succeeds.
- Ensure known lesson page renders from `/topics`.
- Ensure simulation fallback behavior remains non-fatal.

## Actionable Checklist

- Run only relevant checks during iteration.
- Before finalize, run full checks for touched surfaces.
- If skipping a heavy check, document reason in change notes.
