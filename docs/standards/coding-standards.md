# Coding Standards

## Tooling Baseline

- TypeScript strict mode enabled (`tsconfig.json`, target ES2017).
- ESLint uses Next.js core-web-vitals + TypeScript config (`eslint.config.mjs`).
- Prettier rules (`.prettierrc`): semicolons enabled, single quotes, line width 100, tab width 2.
- Path alias: `@/*` maps to `src/*`.
- Next.js 16 with Turbopack, standalone output mode.

## Language And Architecture Rules

- Prefer TypeScript (`.ts`/`.tsx`) for all app logic.
- Use path alias imports via `@/*` for `src/*`.
- Keep concerns separated:
  - `src/domain`: pure logic, deterministic transforms, no framework side effects
  - `src/infra`: adapters for fs/logging/observability/supabase
  - `src/features`: orchestration and caching boundaries
  - `src/shared`: cross-cutting types and error contracts
  - `src/lib`: routing helpers, theme, utilities
  - `src/app`: route composition and HTTP boundary

## API And Error Conventions

- Validate request inputs at route boundary with Zod.
- Throw/use `AppError` for known business and content failures.
- Return error responses through `asErrorEnvelope()` for stable response shape.
- Include or propagate `x-correlation-id` on API responses where implemented.
- Use `logger` from `src/infra/observability/logger.ts` for server-side logging (not raw `console.error`).

## React And Next.js Conventions

- Keep server data retrieval in route/server modules when possible.
- Use client components only when browser APIs or interaction require them.
- For heavy client modules, prefer lazy and viewport-aware loading.
- Avoid broad global state unless explicit cross-page requirement exists.

## Content And Simulation Conventions

- Topic source-of-truth: `src/lib/topic-config.ts` (`TOPICS` record and `TopicSlug` type).
- `src/lib/content.ts` is a convenience re-export of `TOPICS` and `TopicSlug`.
- Route slug mapping: `src/lib/topic-navigation.ts` (`TOPIC_ROUTES`).
- Lesson ordering: `src/lib/topic-navigation.server.ts` (`TOPIC_LESSON_ORDER`, `LESSON_SUMMARIES`).
- Simulation registry ids must align with markdown placeholders:
  - placeholder: `[[simulation my-id]]`
  - registry key: `'my-id': Component`
- Run `npm run check:simulations` after changing placeholders or registries.

## Canvas Charting

- 2D charts use `src/components/ui/canvas-chart.tsx` (pure HTML5 Canvas, no external chart library).
- Heatmaps use `src/components/ui/canvas-heatmap.tsx`.
- Both share theme constants from `src/lib/canvas-theme.ts`.
- Components expose Plotly-compatible trace/layout interfaces for easy data migration.

## Theme System

- Theme is attribute-driven (`data-theme` on `<html>`), not class-driven.
- CSS variables defined in `src/app/globals.css` (light and dark variants).
- `useTheme()` hook from `src/lib/use-theme.ts` reads current theme reactively.
- Canvas components use `getCanvasTheme()` from `src/lib/canvas-theme.ts`.

## Naming Conventions

- files/modules: kebab-case for app/lib/infra files
- React components: PascalCase
- constants: `UPPER_SNAKE_CASE` when immutable config sets
- helper functions: verb-first camelCase

## Testing

- Unit/integration: Vitest (`npm test`)
- End-to-end smoke: Playwright (`npm run test:e2e`)
- Structural integrity: `npm run check:simulations`, `npm run perf:budget`
- Unit tests live with source: `src/domain/**/*.test.ts`, `src/features/**/*.test.ts`, `src/shared/**/*.test.ts`
- E2E tests: `tests/e2e/*.spec.ts`
- Prefer deterministic tests with explicit fixtures.
- Assert contract shape, not incidental implementation details.
- Add regression tests when fixing bugs.

### Required checks by change type

| Change | Checks |
|---|---|
| Content / simulation placeholders | `npm run check:simulations` + `npm test` |
| API route / contract | `npm test` + verify health/ready endpoints |
| Navigation / UI | `npm test` + `npm run test:e2e` |

## Observability

### Signals

- `/api/health`: liveness signal
- `/api/ready`: readiness based on topic index loading
- `logger` (`src/infra/observability/logger.ts`): structured JSON logger (`info`, `warn`, `error`)
- `request-metrics.ts`: logs route, method, status, durationMs, correlationId
- `/api/content` emits `x-correlation-id` for request tracing

### Web Vitals

- Client reports to `/api/analytics/web-vitals`
- Controlled by `NEXT_PUBLIC_ENABLE_WEB_VITALS` (client) and `LOG_WEB_VITALS` (server)

### Performance commands

| Command | Purpose |
|---|---|
| `npm run perf:analyze` | Webpack bundle analyzer |
| `npm run perf:baseline` | Production-like local build + start |
| `npm run perf:budget` | Route JS budget guard |
| `BASE_URL=http://127.0.0.1:3000 npm run perf:load-test` | Content API load probe |

## Do / Don't

### Do

- keep route handlers small and delegate to feature/infra modules
- fail fast on invalid input using typed validation
- isolate failures so one broken module does not cascade

### Don't

- put filesystem reads directly in UI components
- bypass typed error envelope handling for API routes
- add magic strings for topic ids outside central registries
- use raw `console.error` in server routes (use `logger` instead)
