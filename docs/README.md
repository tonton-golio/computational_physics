# Documentation Hub

This directory is the operational map for contributors and autonomous agents working in `computational_physics`.

## Purpose

- explain how the codebase is structured and how data flows through it
- define coding, testing, and contribution standards that match the current repository
- provide safe, repeatable runbooks for release and operations
- provide an explicit autonomous-agent operating contract with escalation paths

## Source-Of-Truth Priority

When documents disagree, use this order:

1. repository and workspace policy files:
   - `../.claw/claw/AGENTS.md`
   - `../../AGENTS.md`
2. executable code and config:
   - `../src/**`
   - `../package.json`
   - `../eslint.config.mjs`
   - `../tsconfig.json`
   - `../.prettierrc`
3. this `docs/` directory

## Documentation Map

- `docs/CONTRIBUTING.md`: contribution workflow and quality gate
- `docs/architecture/codebase-map.md`: module and directory map
- `docs/architecture/request-content-simulation-flow.md`: runtime flow from request to page render
- `docs/standards/coding-standards.md`: TypeScript/React/style conventions
- `docs/standards/testing-standards.md`: unit/e2e/reliability testing conventions
- `docs/guides/content-authoring.md`: lesson markdown authoring conventions
- `docs/guides/simulations.md`: simulation registry and placeholder workflow
- `docs/guides/api-contracts.md`: API endpoint contracts and error envelope
- `docs/agent/operating-model.md`: autonomous execution model and boundaries
- `docs/agent/task-playbooks.md`: repeatable task playbooks for agent execution
- `docs/agent/safety-and-escalation.md`: safety rules and escalation triggers
- `docs/runbooks/release-and-rollback.md`: deployment and rollback checklist
- `docs/operations/slo.md`: service-level objectives
- `docs/operations/incident-response.md`: incident triage and mitigation
- `docs/operations/observability.md`: metrics, logging, and health signals

## Ownership Model

- Platform owner: Anton.
- Any contributor or agent editing docs must keep all command names, paths, and endpoint contracts synchronized with the codebase.
- Docs updates are mandatory for behavior changes in:
  - API contracts
  - content or simulation loading logic
  - operational workflows
  - safety and escalation policies
