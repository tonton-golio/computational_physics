# Release And Rollback Runbook

Related docs:

- `docs/operations/incident-response.md`
- `docs/operations/observability.md`
- `docs/guides/api-contracts.md`

## Release Checklist

1. Ensure CI is green (`lint`, `check:simulations`, `test`, `build`, `perf:budget`).
2. Verify preview deployment for:
   - `/topics`
   - at least one lesson page per major topic
   - `/api/health` and `/api/ready`
3. Validate no spike in API errors from `/api/content`.
4. Promote preview to production.

## Production Verification

1. Open `/topics` and confirm topic navigation cards and links work.
2. Open content containing simulation placeholders and ensure rendering succeeds.
3. Confirm `x-correlation-id` exists on `/api/content` responses.
4. Confirm web-vitals endpoint accepts payloads (`202`).

## Rollback Steps (Vercel)

1. Identify last healthy production deployment in Vercel dashboard.
2. Trigger rollback to that deployment.
3. Verify `/api/ready` returns `200` and simulations appear from `/topics` navigation.
4. Announce incident status and keep degraded deployment locked until root cause is fixed.
