# Content Authoring Guide

## Purpose

Standardize lesson authoring so markdown content loads reliably in pages and APIs.

## Source Location

- Content root: `content/topics/<topic-id>/`
- File formats supported:
  - `<slug>.md`
  - `<slug>.txt`

## Topic Registration

Before adding a new topic directory, update:

- `src/lib/content.ts` (`TOPICS`)
- `src/lib/topic-navigation.ts` (`TOPIC_ROUTES`)
- optional ordering in `src/lib/topic-navigation.server.ts` (`TOPIC_LESSON_ORDER`)

## Lesson File Conventions

- Each lesson should include at least one heading:
  - `# Lesson Title`
- Optional frontmatter block:

```md
---
title: Custom Lesson Title
author: Team
---
```

- If frontmatter `title` is missing, title falls back to first `#` heading, then slug.

## Supported Placeholders

- simulation: `[[simulation simulation-id]]`
- figure: `[[figure figure-id]]`
- code editor: `[[code-editor initial|expected|solution]]`

## Math And Markdown Notes

- Block math:
  - `$$ ... $$`
  - `\\[ ... \\]`
- Inline math:
  - `$ ... $`
  - `\\( ... \\)`
- Keep syntax simple because markdown rendering uses a custom transformer.

## Validation Workflow

After content edits:

1. Run `npm run check:simulations` if any simulation placeholder changed.
2. Run `npm test` for content gateway and domain behavior safety.
3. Open at least one modified lesson route in local dev.

## Do / Don’t

### Do

- keep slugs stable once linked publicly
- keep placeholder ids exact and lowercase where possible
- use one topic directory per content domain id

### Don’t

- invent new placeholder syntax without renderer update
- move topics without updating route and topic registries
- rely on undocumented frontmatter keys for runtime behavior
