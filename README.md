# Koala Brain

> Computational physics you can touch — every equation runs, every model bends, every simulation is yours to break.

**Live at [koalabrain.org](https://koalabrain.org)**

An open, self-evolving learning platform: **10 topics · 114 lessons · 150+ interactive simulations**, built with Next.js 16, React 19, TypeScript (strict), and Tailwind 4, and deployed on Vercel.

## Quickstart

```bash
git clone https://github.com/tonton-golio/computational_physics.git
cd computational_physics
npm install
npm run dev          # → http://localhost:3000
```

No environment variables required — auth and analytics are optional, and the app runs fully without them. Node ≥ 20 (the repo pins Node 22 via `.nvmrc`).

## What's inside

- **114 Markdown lessons** across 10 topics (`content/topics/<topic>/<slug>.md`), with LaTeX via KaTeX and a placeholder syntax for embedding interactives.
- **150+ interactive simulations** on Canvas/p5.js, Three.js (`@react-three/fiber`), and SVG — lazy-loaded through `SimulationHost` with per-registry error isolation.
- **Optional accounts** (Supabase OAuth) powering community suggestions and a contributor leaderboard.
- **Production plumbing** — structured JSON logging, health/readiness probes, correlation IDs, Web Vitals, and per-route JS bundle budgets.

## Project structure

```
src/
  app/         Next.js App Router — pages, API routes, proxy (middleware)
  features/    Content gateway, topic-lessons, simulation system
  domain/      Pure business/domain logic
  infra/       Adapters — file content repo, Supabase clients, observability
  shared/      Cross-cutting contracts and error models
  lib/         Config, markdown rendering, topic navigation, utilities
  components/  React UI — layout, content, visualization, ui
  workers/     Web Workers for heavy computation
content/topics/  Markdown lessons, organized by topic id
```

Lessons embed interactives with a placeholder syntax — `[[simulation <id>]]`, `[[figure <id>]]`, `[[code-toggle …]]` — resolved at render time. The runtime maps each id to a lazily-imported component through a typed registry, isolating any one topic's load failures. The full architecture lives in [`docs/`](docs/).

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Dev server (Turbopack) |
| `npm run build` · `npm start` | Production build · serve |
| `npm test` · `npm run test:e2e` | Unit tests (Vitest) · E2E smoke (Playwright) |
| `npm run lint` | ESLint |
| `npm run check:simulations` | Validate `[[simulation]]` placeholders ↔ registry |
| `npm run perf:budget` · `npm run perf:analyze` | Enforce route bundle budgets · bundle analysis |
| `npm run build:points` | Rebuild the Topics 2D point cloud (needs [`uv`](https://docs.astral.sh/uv/) + `OPENAI_API_KEY`) |

**CI** (Node 22): `lint → check:simulations → test → build → perf:budget`.

## Configuration

Every variable is optional — set only the features you want.

| Variable | Enables |
| --- | --- |
| `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Auth (login, profile, suggestions, leaderboard) |
| `SUPABASE_SERVICE_ROLE_KEY` | Account deletion |
| `NEXT_PUBLIC_ENABLE_WEB_VITALS`, `LOG_WEB_VITALS` | Client / server Web Vitals reporting |
| `OPENAI_API_KEY` | `build:points` topic-cloud generation |

Health and readiness probes are served at `/api/health` and `/api/ready`; content responses carry an `x-correlation-id`.

## Documentation

In-depth references live in [`docs/`](docs/):

- **Architecture** — [codebase map](docs/architecture/codebase-map.md) · [request → content → simulation flow](docs/architecture/request-content-simulation-flow.md)
- **Guides** — [content authoring](docs/guides/content-authoring.md) · [simulations](docs/guides/simulations.md) · [figures](docs/guides/figures.md) · [API contracts](docs/guides/api-contracts.md) · [content style](docs/guides/content-style-guide.md)
- **Standards** — [coding standards](docs/standards/coding-standards.md) · [versioning](docs/versioning.md) · [contributing](docs/CONTRIBUTING.md)

## Autonomous operations

Maintained by [KoalaClaw](https://github.com/koalaclaw69), an autonomous agent that handles content updates, issue triage, and platform upkeep.
