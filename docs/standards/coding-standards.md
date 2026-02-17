# Coding Standards

## Purpose

Document repository coding conventions that are enforceable today.

## Tooling Baseline

- TypeScript strict mode enabled (`tsconfig.json`).
- ESLint uses Next.js core-web-vitals + TypeScript config (`eslint.config.mjs`).
- Prettier rules (`.prettierrc`):
  - semicolons enabled
  - single quotes
  - line width 100
  - tab width 2

## Language And Architecture Rules

- Prefer TypeScript (`.ts`/`.tsx`) for all app logic.
- Use path alias imports via `@/*` for `src/*`.
- Keep concerns separated:
  - `src/domain`: pure logic, deterministic transforms, no framework side effects
  - `src/infra`: adapters for fs/logging/observability
  - `src/features`: orchestration and caching boundaries
  - `src/shared`: cross-cutting types and error contracts
  - `src/app`: route composition and HTTP boundary

## API And Error Conventions

- Validate request inputs at route boundary with Zod.
- Throw/use `AppError` for known business and content failures.
- Return error responses through `asErrorEnvelope()` for stable response shape.
- Include or propagate `x-correlation-id` on API responses where implemented.

## React And Next.js Conventions

- Keep server data retrieval in route/server modules when possible.
- Use client components only when browser APIs or interaction require them.
- For heavy client modules, prefer lazy and viewport-aware loading.
- Avoid broad global state unless explicit cross-page requirement exists.

## Content And Simulation Conventions

- Topic source-of-truth key list lives in `src/lib/content.ts`.
- Route slug mapping lives in `src/lib/topic-navigation.ts`.
- Simulation registry ids must align with markdown placeholders:
  - placeholder: `[[simulation my-id]]`
  - registry key: `'my-id': Component`
- Run `npm run check:simulations` after changing placeholders or registries.

## Naming Conventions

- files/modules: kebab-case for app/lib/infra files
- React components: PascalCase
- constants: `UPPER_SNAKE_CASE` when immutable config sets
- helper functions: verb-first camelCase

## Do / Don’t

### Do

- keep route handlers small and delegate to feature/infra modules
- fail fast on invalid input using typed validation
- isolate failures so one broken module does not cascade

### Don’t

- put filesystem reads directly in UI components
- bypass typed error envelope handling for API routes
- add magic strings for topic ids outside central registries
