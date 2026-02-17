# Agent Task Playbooks

## Purpose

Provide repeatable playbooks for frequent autonomous tasks.

## Playbook: Fix Content Rendering Issue

1. Reproduce on lesson route under `/topics/<topic>/<slug>`.
2. Check route mapping:
   - `src/lib/topic-navigation.ts`
3. Check lesson existence and parse:
   - `content/topics/<topic-id>/<slug>.md`
   - `src/infra/content/file-content-repository.ts`
4. If placeholders involved, validate renderer rules:
   - `src/components/content/MarkdownContent.tsx`
5. Run:
   - `npm test`
   - `npm run check:simulations` (if simulation placeholders affected)
6. Update related docs guide if behavior changed.

## Playbook: Add Or Repair Simulation

1. Identify failing or missing simulation id from markdown.
2. Confirm id key exists in a `*Simulations.tsx` registry.
3. Confirm registry group exists in `src/features/simulation/simulation-manifest.ts`.
4. Validate runtime behavior in lesson page and fallback path.
5. Run:
   - `npm run check:simulations`
   - `npm test`
6. Document conventions in `docs/guides/simulations.md` when changed.

## Playbook: API Contract Change

1. Update route handler in `src/app/api/.../route.ts`.
2. Keep input validation explicit with Zod.
3. Keep or intentionally evolve `asErrorEnvelope()` contract.
4. Ensure request metrics/correlation behavior remains consistent.
5. Run:
   - `npm run lint`
   - `npm test`
6. Update `docs/guides/api-contracts.md` in same change.

## Playbook: Reliability Incident Response

1. Confirm signal:
   - `/api/health`
   - `/api/ready`
   - recent request metrics logs
2. Classify impact (content API, simulation render, route availability).
3. Mitigate:
   - rollback if needed using `docs/runbooks/release-and-rollback.md`
4. Add regression test or guardrail script update.
5. Record retrospective and next-step action.

## Playbook: Weekly Subject Addition

1. Read queue in `.claw/workflows/subject_queue.md`.
2. Execute `.claw/workflows/weekly_subject_scheduler.md`.
3. Execute `.claw/workflows/add_new_subject.md`.
4. Update topic metadata, route mapping, and content tree.
5. Validate placeholders and tests before publish.

## Completion Checklist (all playbooks)

- Affected checks run and pass.
- Docs updated for any changed behavior or contract.
- Safety and escalation policy respected.
