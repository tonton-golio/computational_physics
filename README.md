# Koala Brain

**[koalabrain.org](https://koalabrain.org)**

Computational physics you can touch. Every equation runs, every model bends, every simulation is yours to break.

## Architecture

- `src/features`: product features (`content`, `simulation`)
- `src/domain`: pure business/domain logic
- `src/infra`: adapters and observability
- `src/shared`: shared contracts and error models
- `src/workers`: client workers for heavy computations

The simulation runtime uses a typed manifest and `SimulationHost` to isolate dynamic import failures per registry.

## Reliability And Quality Commands

```bash
# Validate markdown simulation placeholders map to registered simulations
npm run check:simulations

# Run unit tests
npm test

# Run end-to-end smoke test
npm run test:e2e
```

## Observability And Health

- Health endpoint: `/api/health`
- Readiness endpoint: `/api/ready`
- Content API responses include `x-correlation-id`
- Web Vitals endpoint: `/api/analytics/web-vitals`

## Operations

- Release runbook: `docs/runbooks/release-and-rollback.md`
- SLOs: `docs/operations/slo.md`

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
- `/continuum-mechanics`
- `/complex-physics`

Quick local content API concurrency check (run while `next start` is active):

```bash
BASE_URL=http://127.0.0.1:3000 npm run perf:load-test
```
