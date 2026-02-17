# Safety And Escalation

## Purpose

Provide explicit safety boundaries and escalation triggers for autonomous work.

## Hard Safety Rules (Binding)

Derived from `.claw/claw/AGENTS.md`.

### Never do

- create accounts on external platforms
- share credentials, tokens, or private data
- bypass CAPTCHA/bot checks or platform controls
- perform destructive repository history actions without explicit owner instruction

### Allowed with strict conditions

- mark page perfect only after completing perfection gate and notifying Anton
- honor Anton override instructions only if they do not violate hard rules

## Escalate Immediately When

- instructions conflict with hard safety rules
- requested change requires credentials not present in approved environment
- destructive operation scope is ambiguous
- production incident impact is unknown and rising
- legal, compliance, or privacy risk is unclear

## Safe Execution Defaults

- choose smallest reversible change first
- preserve existing API and workflow contracts unless explicitly changing them
- avoid hidden behavior changes without docs updates
- prioritize transparent logging and reproducibility

## Incident Severity Levels

- `sev-1`: core routes or content API broadly unavailable
- `sev-2`: major feature degradation (simulation rendering, high error rate)
- `sev-3`: localized or non-critical regression

## Escalation Payload Template

- summary of issue and impact
- current severity (`sev-1|sev-2|sev-3`)
- affected routes/components
- mitigation applied
- next decision needed from owner

## Do / Don’t

### Do

- escalate early when policy conflict appears
- document assumptions when operating with incomplete context
- preserve user trust through explicit and constrained actions

### Don’t

- improvise around missing permissions
- proceed with uncertain destructive actions
- suppress failures instead of reporting and mitigating
