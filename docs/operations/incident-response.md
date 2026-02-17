# Incident Response Runbook

## Purpose

Define a practical response path for service degradation or outage.

## Scope

User-critical paths:

- `/topics` and lesson pages
- `/api/content`
- simulation placeholder rendering

## Triage Steps

1. Confirm incident with independent checks:
   - `GET /api/health`
   - `GET /api/ready`
   - open `/topics` and one representative lesson route
2. Determine blast radius:
   - all topics vs one topic
   - all simulations vs subset
   - API latency vs API failure
3. Classify severity (`sev-1`, `sev-2`, `sev-3`).

## Mitigation Options

- revert or rollback recent deployment using `docs/runbooks/release-and-rollback.md`
- disable or isolate failing simulation/module path where feasible
- ship minimal hotfix with targeted tests

## Required Communication Payload

- incident start timestamp (UTC)
- impacted user paths
- severity and confidence
- mitigation chosen and ETA for reassessment

## Recovery Verification

- `/api/ready` returns `200`.
- `/api/content` serves expected payload and headers.
- at least one simulation placeholder renders without error.
- post-mitigation checks pass:
  - `npm run check:simulations`
  - `npm test`

## Post-Incident Actions

- add regression test or automated check for root cause class
- update relevant runbook/docs
- log retrospective via calendar workflow
