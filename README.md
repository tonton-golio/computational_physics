This is a Next.js computational physics platform with interactive simulations, content-driven lessons.
## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Architecture

- `src/features`: product features (`content`, `simulation`)
- `src/domain`: pure business/domain logic
- `src/infra`: adapters and observability
- `src/shared`: shared contracts and error models
- `src/workers`: client workers for heavy computations

The simulation runtime now uses a typed manifest and `SimulationHost` to isolate dynamic import failures per registry.

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

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy On Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

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
