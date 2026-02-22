# Koala Brain

**[koalabrain.org](https://koalabrain.org)**

Computational physics you can touch. Every equation runs, every model bends, every simulation is yours to break.

## Architecture

Next.js 16 (React 19, TypeScript strict, Tailwind 4) educational platform with 10 topics, ~114 lessons, and 190+ interactive simulations. Deployed on Vercel (standalone output).

```
src/
  app/            # Next.js App Router — pages, API routes, middleware
  features/       # Feature modules (content gateway, topic-lessons, simulation system)
  domain/         # Pure business/domain logic
  infra/          # Adapters (file-content-repository, supabase clients, observability)
  shared/         # Shared contracts and error models
  lib/            # Config, utilities, markdown rendering, topic navigation
  components/     # React components (layout, content, visualization, ui)
  workers/        # Web Workers for heavy computation
content/topics/   # Markdown lesson files organized by topic ID
```

The simulation runtime uses a typed manifest and `SimulationHost` to isolate dynamic import failures per registry. Simulations use Canvas/p5.js, Three.js (@react-three/fiber), or SVG with a slot-based UI model (`SimulationPanel`, `SimulationMain`, `SimulationSettings`, `SimulationConfig`, `SimulationResults`, `SimulationAux`).

## Reliability And Quality Commands

```bash
# Validate markdown simulation placeholders map to registered simulations
npm run check:simulations

# Run unit tests
npm test

# Run end-to-end smoke test
npm run test:e2e

# ESLint
npm run lint
```

CI pipeline (Node 22): lint → check:simulations → test → build → perf:budget

## Observability And Health

- Health endpoint: `/api/health`
- Readiness endpoint: `/api/ready`
- Content API responses include `x-correlation-id`
- Web Vitals endpoint: `/api/analytics/web-vitals`
- Structured JSON logging via `src/infra/observability/logger.ts`
- Request metrics with correlation IDs via `src/infra/observability/request-metrics.ts`

## Topic Points

Generate the 2D points JSON used by the Topics visualization. This embeds each sub-topic via OpenAI and projects the vectors to 2D with t-SNE.

**Prerequisites:**

- [`uv`](https://docs.astral.sh/uv/) installed (`brew install uv`)
- `OPENAI_API_KEY` set in `.env` at the project root

```bash
npm run build:points
```

Output: `public/data/topic-points.json`

Optional flags (pass after `--`):

```bash
npm run build:points -- --output path/to/out.json --batch-size 32 --max-chars 4000
```

## Performance Baseline And Monitoring

- Web Vitals are reported from the client via `useReportWebVitals` to `/api/analytics/web-vitals`.
- Enable reporting locally by setting `NEXT_PUBLIC_ENABLE_WEB_VITALS=1`.
- Enable server-side log output in production with `LOG_WEB_VITALS=1`.

Useful commands:

```bash
# Build with bundle analysis report (see ./analyze/client.html)
npm run perf:analyze

# Production-like local run for Lighthouse/WebPageTest style checks
npm run perf:baseline

# Enforce route-level JS bundle budgets after build
npm run perf:budget
```

Recommended baseline routes:

- `/topics`
- `/topics/continuum-mechanics`
- `/topics/complex-physics`

Quick local content API concurrency check (run while `next start` is active):

```bash
BASE_URL=http://127.0.0.1:3000 npm run perf:load-test
```
