# Observability Guide

## Purpose

Describe what signals exist today and how to use them for debugging and reliability.

## Existing Signals

### Health and readiness endpoints

- `/api/health`: liveness signal
- `/api/ready`: readiness based on topic index loading

### Request metrics

- `src/infra/observability/request-metrics.ts` logs:
  - route
  - method
  - status
  - durationMs
  - correlationId

### Correlation id

- `/api/content` emits `x-correlation-id` for request tracing.

### Web Vitals ingestion

- client reports to `/api/analytics/web-vitals`
- controlled by:
  - `NEXT_PUBLIC_ENABLE_WEB_VITALS`
  - `LOG_WEB_VITALS`

## Local Debug Workflow

1. Start app: `npm run dev`.
2. Trigger affected endpoint or route.
3. Capture readiness/health output.
4. Inspect logs for request metric anomalies.
5. For page performance checks:
   - enable `NEXT_PUBLIC_ENABLE_WEB_VITALS=1`
   - optionally set `LOG_WEB_VITALS=1` for server-side metric logs

## Performance And Load Checks

- `npm run perf:analyze`: build analyzer output
- `npm run perf:baseline`: production-like local run
- `npm run perf:budget`: route JS budget guard
- `BASE_URL=http://127.0.0.1:3000 npm run perf:load-test`: content API load probe

## Observability Gaps To Track

- no explicit alerting rules in-repo
- no central incident timeline artifact in-repo
- limited endpoint-level SLI breakdown beyond current logs

## Actionable Checklist

- For every incident, preserve a correlation id sample.
- For API regressions, compare status and duration trends before/after fix.
- Keep this guide updated when new metrics or logging sinks are introduced.
