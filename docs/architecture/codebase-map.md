# Codebase Map

## Purpose

Provide a concrete directory and module map so autonomous execution can locate the right file on first pass.

## Top-Level Structure

- `content/topics/`: markdown lessons grouped by topic slug
- `src/app/`: Next.js App Router pages and API routes
- `src/features/`: feature-level orchestration (`content`, `simulation`)
- `src/infra/`: file repositories and observability adapters
- `src/domain/`: pure domain transforms and models
- `src/shared/`: shared types and error contracts
- `src/components/`: UI and visualization component registries
- `src/lib/`: routing and content helper functions
- `scripts/`: validation and performance utility scripts
- `.claw/`: autonomous workflow instructions and state artifacts

## Runtime-Critical Files

### Content retrieval

- `src/app/api/content/route.ts`: public content API contract
- `src/features/content/content-gateway.ts`: topic and lesson retrieval orchestration with cache fallback
- `src/infra/content/file-content-repository.ts`: filesystem read and frontmatter parsing
- `src/lib/content.ts`: `TOPICS` metadata registry and topic-level helpers

### Page rendering

- `src/app/topics/[topic]/[slug]/page.tsx`: lesson page route, static params, lesson nav
- `src/components/content/MarkdownContent.tsx`: markdown conversion and placeholder extraction
- `src/lib/topic-navigation.ts`: route slug to content id mapping
- `src/lib/topic-navigation.server.ts`: topic-specific lesson ordering rules

### Simulation runtime

- `src/features/simulation/simulation-manifest.ts`: lazy registry resolution by simulation id
- `src/features/simulation/SimulationHost.tsx`: viewport-aware simulation loading with fallback graph
- `src/components/visualization/*Simulations.tsx`: topic simulation registries

### Reliability and observability

- `src/app/api/health/route.ts`: liveness endpoint
- `src/app/api/ready/route.ts`: readiness endpoint (content index probe)
- `src/app/api/analytics/web-vitals/route.ts`: client vitals ingestion
- `src/infra/observability/request-metrics.ts`: request metric event emission
- `src/shared/errors/app-error.ts`: stable API error envelope

## Agent-Facing Control Plane

- `.claw/claw/AGENTS.md`: operating policy and priority queue
- `.claw/workflows/`: executable runbooks for recurring operations
- `.claw/workflows/page_touch_manifest.json`: staleness and quality pass tracking
- `.claw/workflows/subject_queue.md`: weekly subject scheduling source

## Actionable Checklist

- Before edits, identify whether the task belongs to `app`, `features`, `infra`, `domain`, `shared`, or `.claw`.
- For page/content bugs, inspect `topic-navigation` and `file-content-repository` before changing UI code.
- For simulation placeholder failures, run `npm run check:simulations` and inspect topic registries.
- For API changes, update `docs/guides/api-contracts.md` in the same change.
