# Service Level Objectives

Related docs:

- `docs/operations/observability.md`
- `docs/operations/incident-response.md`
- `docs/runbooks/release-and-rollback.md`

## Scope

Applies to user-critical paths:
- `/topics` route
- topic lesson content retrieval via `/api/content`
- simulation component mounting from markdown placeholders

## SLO Targets

- Availability: 99.9% monthly for `/api/content`
- Latency: p95 `/api/content` under 500ms for cached reads
- Render reliability: 99.5% successful simulation mounts on placeholder render attempts

## Error Budget Policy

- If monthly error budget burn exceeds 50%, freeze non-critical feature work.
- Prioritize reliability fixes, observability gaps, and regression tests.

## Measurement Inputs

- Request metrics logs from API routes
- Web-vitals and client-side simulation error counters
- CI stability and production incident reports
