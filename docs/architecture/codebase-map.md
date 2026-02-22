# Codebase Map

## Purpose

Provide a concrete directory and module map so contributors and autonomous agents can locate the right file on first pass.

## Top-Level Structure

```
computational_physics/
  content/topics/       markdown lessons grouped by topic slug (10 topics, ~114 files)
  docs/                 project documentation
  scripts/              validation, performance, and build utility scripts
  src/                  application source code
  public/               static assets (images, figures, data files)
  .claw/                autonomous workflow instructions, skills, and state artifacts
```

## src/ Directory Map

```
src/
  app/                  Next.js App Router pages and API routes
    api/
      analytics/web-vitals/route.ts   client vitals ingestion (POST)
      auth/callback/route.ts          OAuth code exchange (GET)
      auth/delete-account/route.ts    account deletion (DELETE)
      content/route.ts                content API (GET)
      content/export/route.ts         content export (GET)
      health/route.ts                 liveness check (GET)
      ready/route.ts                  readiness check (GET)
      suggestions/route.ts            user suggestions (POST)
    leaderboard/page.tsx
    login/page.tsx
    profile/                          page.tsx, ProfileActions.tsx, SuggestionsTable.tsx
    topics/page.tsx                   topic grid / point cloud
    topics/[topic]/page.tsx           topic landing page
    topics/[topic]/[slug]/page.tsx    lesson page

  auth-middleware.ts    Supabase auth middleware (protected routes, redirects)

  components/
    content/            MarkdownContent.tsx, ExportPdfButton.tsx
    effects/            AntigravityClient.tsx (homepage particle effect)
    layout/             Header, AuthButton, ThemeToggle, LeaderboardButton,
                        SuggestionBox, TopicSidebar, CollapsibleTopicLayout
    performance/        WebVitalsReporter.tsx
    topics/             TopicsSearchGrid.tsx (point cloud + search)
    ui/                 button, card, canvas-chart, canvas-heatmap, code-editor,
                        CodeToggleBlock, figure-lightbox, fullscreen-button,
                        simulation-main, simulation-panel, label, slider, table
    visualization/
      *Simulations.tsx  11 topic simulation registries
      InteractiveGraph.tsx              fallback graph component
      advanced-deep-learning/           16 simulation components
      applied-machine-learning/         23 components + ml-utils.ts
      applied-statistics/               23 components
      complex-physics/                  20 components
      continuum-mechanics/              18 components
      dynamical-models/                 21 components
      eigenvalue/                       10 components + eigen-utils.ts
      inverse-problems/                 16 components
      nonlinear-equations/              2 components (Himmelblau2D, Newton1D)
      online-reinforcement/             21 components
      quantum-optics/                   10 components
      scientific-computing/             12 components

  domain/content/       models.ts (pure transforms), models.test.ts
  features/
    content/            content-gateway.ts (orchestration + cache),
                        topic-lessons.ts (ordering, landing pages, lesson listing), test
    simulation/         SimulationHost.tsx, simulation-manifest.ts,
                        simulation-worker.client.ts
  infra/
    content/            file-content-repository.ts (filesystem read + parse)
    observability/      logger.ts, request-metrics.ts
    supabase/           client.ts (browser), server.ts (server with cookie sync)
  lib/
    canvas-theme.ts     shared dark/light theme for canvas components
    chart-colors.ts     color palette constants
    content.ts          re-exports TOPICS and TopicSlug from topic-config
    figure-definitions.ts   aggregated figure metadata
    markdown-to-html.ts     markdown + LaTeX + placeholder conversion
    simulation-descriptions.ts  aggregated simulation descriptions
    topic-config.ts     TOPICS record (10 topics) and TopicSlug type
    topic-navigation.ts route slug to content id mapping
    topic-navigation.server.ts  lesson ordering, summaries, landing pages
    use-theme.ts        React hook for data-theme attribute
    utils.ts            cn() utility (clsx + tailwind-merge)
  shared/
    errors/             AppError, ErrorEnvelope, asErrorEnvelope()
    types/
      content.ts        TopicDefinition, TopicIndex, ContentDocument
      simulation.ts     SimulationComponent, SimulationDefinition,
                        LotkaVolterraParams/Result, MDPSimulationParams/Result
  workers/simulation/   lotka-volterra.worker.ts, mdp-simulation.worker.ts
```

## Runtime-Critical File Index

### Content retrieval

| File | Role |
|---|---|
| `src/lib/topic-config.ts` | `TOPICS` metadata registry (titles, colors, difficulty) and `TopicSlug` type |
| `src/lib/content.ts` | Re-exports `TOPICS` and `TopicSlug` from topic-config |
| `src/lib/topic-navigation.ts` | Route slug to content id mapping (`TOPIC_ROUTES`) |
| `src/lib/topic-navigation.server.ts` | Lesson ordering config (`TOPIC_LESSON_ORDER`), summaries (`LESSON_SUMMARIES`), landing page priority |
| `src/features/content/topic-lessons.ts` | Ordering functions (`getOrderedLessonSlugs`, `getLessonsForTopic`, `getLandingPageSlug`) |
| `src/infra/content/file-content-repository.ts` | Filesystem read, frontmatter parse, Zod validation |
| `src/features/content/content-gateway.ts` | Cached topic/lesson retrieval orchestration |
| `src/app/api/content/route.ts` | Public content API endpoint |

### Page rendering

| File | Role |
|---|---|
| `src/app/topics/[topic]/[slug]/page.tsx` | Lesson page route with static params and lesson nav |
| `src/components/content/MarkdownContent.tsx` | Markdown to HTML + placeholder extraction and React rendering |
| `src/lib/markdown-to-html.ts` | Markdown/LaTeX/placeholder transformer |

### Simulation runtime

| File | Role |
|---|---|
| `src/features/simulation/simulation-manifest.ts` | Lazy registry resolution by simulation id (11 registry groups) |
| `src/features/simulation/SimulationHost.tsx` | Viewport-aware simulation loading with `InteractiveGraph` fallback |
| `src/components/visualization/*Simulations.tsx` | Per-topic simulation registries (11 total) |

### Canvas charting

| File | Role |
|---|---|
| `src/components/ui/canvas-chart.tsx` | Pure-canvas 2D chart (scatter, line, bar, histogram) |
| `src/components/ui/canvas-heatmap.tsx` | Pure-canvas heatmap with colorbar |
| `src/lib/canvas-theme.ts` | Shared dark/light theme constants for both canvas components |

### Auth and middleware

| File | Role |
|---|---|
| `src/auth-middleware.ts` | Supabase auth middleware (route protection, redirects) |
| `src/infra/supabase/client.ts` | Browser Supabase client |
| `src/infra/supabase/server.ts` | Server Supabase client (async cookies) |

### Observability

| File | Role |
|---|---|
| `src/app/api/health/route.ts` | Liveness endpoint |
| `src/app/api/ready/route.ts` | Readiness endpoint (content index probe) |
| `src/infra/observability/logger.ts` | Structured JSON logger (`logger.info/warn/error`) |
| `src/infra/observability/request-metrics.ts` | Request metric + correlation id emission |
| `src/shared/errors/app-error.ts` | Typed error codes + stable API error envelope |

## Topics (10 total)

All topics have matching route slugs and content IDs.

| Content ID | Lessons | Difficulty |
|---|---|---|
| `applied-statistics` | 14 | Beginner |
| `applied-machine-learning` | 8 | Intermediate |
| `dynamical-models` | 10 | Beginner |
| `scientific-computing` | 11 | Intermediate |
| `complex-physics` | 13 | Intermediate |
| `continuum-mechanics` | 16 | Expert |
| `inverse-problems` | 8 | Advanced |
| `quantum-optics` | 14 | Expert |
| `online-reinforcement-learning` | 11 | Intermediate |
| `advanced-deep-learning` | 9 | Advanced |

## Agent-Facing Control Plane

- `.claw/AGENTS.md`: operating policy and priority queue
- `.claw/workflows/`: executable runbooks for recurring operations (add_new_subject, process_suggestion, post_to_x)
- `.claw/skills/`: reusable agent skills (kcs-checkpoint, validate-suggestion, update-point-cloud, etc.)

## Actionable Checklist

- Before edits, identify whether the task belongs to `app`, `features`, `infra`, `domain`, `shared`, or `.claw`.
- For page/content bugs, inspect `topic-navigation` and `file-content-repository` before changing UI code.
- For simulation placeholder failures, run `npm run check:simulations` and inspect topic registries.
- For API changes, update `docs/guides/api-contracts.md` in the same change.
- Topic metadata lives in `src/lib/topic-config.ts` (the authoritative source); `src/lib/content.ts` is a convenience re-export.
