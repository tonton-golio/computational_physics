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
npm run perf:baseline    # Production-like local build + start for Lighthouse checks
npm run perf:load-test   # Content API load testing (run while next start is active)
npm run build:points     # Generate topic cloud JSON (requires uv + OPENAI_API_KEY)
```

CI pipeline (Node 22): lint → check:simulations → test → build → perf:budget

**Before committing, always run `npm test` and `npm run lint` and confirm they pass.**

## Architecture

Next.js 16 (React 19, TypeScript strict, Tailwind 4) educational platform for computational physics. Deployed on Vercel (standalone output).

### Layer structure

```
src/
  app/            # Next.js App Router — pages, API routes, middleware
  features/       # Feature modules (content gateway, topic-lessons, simulation system)
  domain/         # Pure business logic and models
  infra/          # Adapters (file-content-repository, supabase clients, observability)
  shared/         # Cross-cutting types (content, simulation) and AppError
  lib/            # Config, utilities, markdown rendering, topic navigation
  components/     # React components (layout, content, visualization, ui)
  workers/        # Web Workers for heavy computation
content/topics/   # Markdown lesson files organized by topic ID
```

### Content system

- 10 topics, ~114 markdown lessons in `content/topics/{topicId}/{slug}.md`
- Lessons have optional YAML frontmatter; title extracted from `# Heading` or frontmatter
- Content parsed by `src/infra/content/file-content-repository.ts` (Zod-validated)
- Lesson ordering and summaries configured in `src/lib/topic-navigation.server.ts`
- Ordering functions (`getOrderedLessonSlugs`, `getLessonsForTopic`, `getLandingPageSlug`) live in `src/features/content/topic-lessons.ts`
- Served via `src/features/content/content-gateway.ts` using `unstable_cache`
- API route: `GET /api/content?topic=X&slug=Y` (cached 3600s, includes `x-correlation-id`)

### Simulation system

- Custom placeholder syntax in markdown: `[[simulation id]]`, `[[figure id]]`, `[[code-toggle python|pseudo]]`
- `src/components/content/MarkdownContent.tsx` parses HTML and renders placeholders as React components
- `src/features/simulation/simulation-manifest.ts` — registry-based dynamic imports, organized by topic (10 registry groups with LRU caching and per-registry error isolation)
- `src/features/simulation/SimulationHost.tsx` — lazy-loads simulations via Intersection Observer (render at 200px, prefetch at 1200px with `requestIdleCallback`); wraps simulations in `SimulationFullscreenProvider` for fullscreen support
- 190+ simulation components in `src/components/visualization/` organized by topic subdirectory
- Simulations use Canvas/p5.js, Three.js (@react-three/fiber), or SVG
- Simulation UI uses a slot model: `SimulationPanel`, `SimulationMain`, `SimulationSettings`, `SimulationConfig`, `SimulationResults`, `SimulationAux` (see `docs/guides/simulations.md`)

### Key config files

- `src/lib/topic-config.ts` — master topic metadata (id, title, difficulty, color, relatedTopics)
- `src/lib/topic-navigation.ts` — route slug ↔ content ID mapping, URL helpers
- `src/lib/topic-navigation.server.ts` — lesson ordering config (`TOPIC_LESSON_ORDER`), landing page priority, lesson summaries (`LESSON_SUMMARIES`)
- `src/features/content/topic-lessons.ts` — ordering functions, landing page resolution, lesson listing
- `src/lib/figure-definitions.ts` — figure registry (image/video sources and captions)
- `src/lib/simulation-descriptions.ts` — simulation title registry

### Auth system

- Supabase for authentication (OAuth flow)
- Supabase clients: `src/infra/supabase/client.ts` (browser), `src/infra/supabase/server.ts` (server with cookie sync)
- Auth middleware: `src/auth-middleware.ts` — protects `/profile`, redirects unauthenticated users
- Pages: `/login`, `/profile`, `/leaderboard`
- API routes: `GET /api/auth/callback` (OAuth code exchange), `POST /api/auth/delete-account`
- Auth is optional — the app works without Supabase credentials configured

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
- `docs/guides/figures.md` — how to add static figures (images/videos) with lightbox
- `docs/standards/coding-standards.md` — code style, naming, and patterns used in this repo
- `docs/versioning.md` — version string format and bump rules
- `docs/CONTRIBUTING.md` — contribution workflow and PR expectations

### Environment variables

```
NEXT_PUBLIC_SUPABASE_URL          # Supabase project URL (optional — auth disabled without it)
NEXT_PUBLIC_SUPABASE_ANON_KEY     # Supabase anon key
SUPABASE_SERVICE_ROLE_KEY         # Supabase service role key (account deletion)
NEXT_PUBLIC_ENABLE_WEB_VITALS=0   # Set to 1 to report Web Vitals from client
LOG_WEB_VITALS=0                  # Set to 1 for server-side Web Vitals logging
OPENAI_API_KEY                    # For build:points (topic cloud generation)
BASE_URL                          # For perf:load-test
```

## Adding content

1. Create `content/topics/{topicId}/{slug}.md` with optional YAML frontmatter
2. Add the slug to `TOPIC_LESSON_ORDER` in `src/lib/topic-navigation.server.ts`
3. Add a summary to `LESSON_SUMMARIES` in the same file
4. Add any `[[simulation id]]` placeholders and register them in the topic `*Simulations.tsx` registry
5. Add simulation components in `src/components/visualization/{topic}/`
6. Run `npm run check:simulations` to validate placeholders

## Build:points (topic cloud visualization)

Requires `uv` and `OPENAI_API_KEY` in `.env`. Embeds sub-topics via OpenAI → t-SNE projection → `public/data/topic-points.json`. Caches embeddings, so only new/changed lessons trigger API calls.

```bash
npm run build:points
```
