# Autonomous Agent Operating Model

## Purpose

Define how an autonomous agent should operate in this repository while respecting hard constraints and priorities.

## Authority And Scope

- Primary policy source: `.claw/claw/AGENTS.md`.
- Workspace boundary: only work in `computational_physics/`.
- Mission context: increase platform quality, reliability, and operational responsiveness.

## Priority Queue

Execution priority follows `.claw/claw/AGENTS.md`:

1. Anton Gmail instructions
2. Active incidents and bug reports
3. Hourly stale-page improver
4. Community response and engagement
5. Weekly subject addition
6. Backlog hygiene and quality upgrades

## Execution Principles

- Prefer direct execution with verifiable outcomes.
- Keep tasks small, testable, and documented.
- Use existing workflow files in `.claw/workflows/` as executable checklists.
- Treat docs + code as one delivery unit for behavioral changes.

## Decision Policy

- If a requested action conflicts with hard safety rules, refuse and escalate.
- If requirements are ambiguous and risk is non-trivial, request clarification.
- If issue is routine and within policy, execute autonomously and log outcome.

## Required Artifacts After Work

- changed code/docs (if applicable)
- workflow state updates (when applicable)
- calendar retrospective entry (per `.claw/workflows/calendar_logging.md`)

## Do / Don’t

### Do

- follow source-of-truth documents over inferred behavior
- preserve API contracts unless intentionally updated
- keep reliability checks in the loop for production-facing changes

### Don’t

- operate outside `computational_physics/`
- bypass explicit safety boundaries
- mark pages perfect without executing the perfection gate
