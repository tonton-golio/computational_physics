# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev              # Dev server (Turbopack)
npm run build            # Production build
npm run lint             # ESLint
npm test                 # Vitest unit tests
npm run test:watch       # Vitest watch mode
npm run test:e2e         # Playwright E2E tests
npm run check:simulations # Validate markdown [[simulation]] placeholders match registered simulations
npm run perf:budget      # Enforce route-level JS bundle budgets (run after build)
npm run perf:analyze     # Webpack bundle analysis (outputs ./analyze/client.html)
```

CI pipeline (Node 22): lint → check:simulations → test → build → perf:budget

**Before committing, always run `npm test` and `npm run lint` and confirm they pass.**

## Architecture

Next.js 16 (React 19, TypeScript strict, Tailwind 4) educational platform for computational physics. Deployed on Vercel (standalone output).

### Layer structure

```
src/
  app/            # Next.js App Router — pages and API routes
  features/       # Feature modules (content-gateway, simulation system)
  domain/         # Pure business logic and models
  infra/          # Adapters (file-content-repository) and observability
  shared/         # Cross-cutting types (content, simulation) and AppError
  lib/            # Config, utilities, markdown rendering
  components/     # React components (layout, content, visualization, ui)
  workers/        # Web Workers for heavy computation
content/topics/   # Markdown lesson files organized by topic ID
```

### Content system

- 10 topics, ~123 markdown lessons in `content/topics/{topicId}/{slug}.md`
- Lessons have optional YAML frontmatter; title extracted from `# Heading` or frontmatter
- Content parsed by `src/infra/content/file-content-repository.ts` (Zod-validated)
- Served via `src/features/content/content-gateway.ts` using `unstable_cache`
- API route: `GET /api/content?topic=X&slug=Y` (cached 3600s, includes `x-correlation-id`)

### Simulation system

- Custom placeholder syntax in markdown: `[[simulation id]]`, `[[figure id]]`, `[[code-toggle python|pseudo]]`
- `src/components/content/MarkdownContent.tsx` parses HTML and renders placeholders as React components
- `src/features/simulation/simulation-manifest.ts` — registry-based dynamic imports, organized by topic (10 registry groups with LRU caching and per-registry error isolation)
- `src/features/simulation/SimulationHost.tsx` — lazy-loads simulations via Intersection Observer (render at 200px, prefetch at 1200px with `requestIdleCallback`)
- 100+ simulation components in `src/components/visualization/` organized by topic subdirectory
- Simulations use Canvas/p5.js, Three.js (@react-three/fiber), or SVG

### Key config files

- `src/lib/topic-config.ts` — master topic metadata (id, title, difficulty, color, relatedTopics)
- `src/lib/topic-navigation.ts` — route slug ↔ content ID mapping, URL helpers
- `src/lib/topic-navigation.server.ts` — lesson ordering, landing page priority, lesson summaries
- `src/lib/figure-definitions.ts` — figure registry (image/video sources and captions)
- `src/lib/simulation-descriptions.ts` — simulation title registry

### Path alias

`@/*` maps to `./src/*` (tsconfig paths).

## Documentation

The `docs/` folder contains detailed reference material. Read the relevant files before making non-trivial changes.

- `docs/architecture/codebase-map.md` — full codebase map and module responsibilities
- `docs/architecture/request-content-simulation-flow.md` — request lifecycle from URL to rendered simulation
- `docs/guides/content-authoring.md` — how to write and structure lesson markdown
- `docs/guides/simulations.md` — how to build and register simulation components
- `docs/guides/api-contracts.md` — API route schemas, query params, and response shapes
- `docs/guides/content-style-guide.md` — tone, formatting, and writing conventions for lesson content
- `docs/standards/coding-standards.md` — code style, naming, and patterns used in this repo
- `docs/CONTRIBUTING.md` — contribution workflow and PR expectations

## Adding content

1. Create `content/topics/{topicId}/{slug}.md` with optional YAML frontmatter
2. Add any `[[simulation id]]` placeholders and register them in `simulation-manifest.ts`
3. Add simulation components in `src/components/visualization/{topic}/`
4. Update `TOPIC_LESSON_ORDER` in `topic-navigation.server.ts` if ordering matters
5. Run `npm run check:simulations` to validate placeholders

## Build:points (topic cloud visualization)

Requires `uv` and `OPENAI_API_KEY` in `.env`. Embeds sub-topics via OpenAI → t-SNE projection → `public/data/topic-points.json`.

```bash
npm run build:points
```
