# Content Authoring Guide

## Purpose

Standardize lesson authoring so markdown content loads reliably in pages and APIs.

## Source Location

- Content root: `content/topics/<topic-id>/`
- File formats supported: `<slug>.md` or `<slug>.txt`

## Topic Registration

Before adding a new topic directory, update these files:

1. `src/lib/topic-config.ts` - add entry to `TOPICS` record (title, description, difficulty, color, relatedTopics)
2. `src/lib/topic-navigation.ts` - add entry to `TOPIC_ROUTES` array (routeSlug and contentId must match)
3. `src/lib/topic-navigation.server.ts` - add entries to:
   - `TOPIC_LESSON_ORDER` (ordered array of lesson slugs)
   - `LESSON_SUMMARIES` (one-line descriptions for each lesson)

Note: `src/lib/content.ts` is a convenience re-export of `TOPICS` and `TopicSlug` from `topic-config.ts`. The authoritative source is `topic-config.ts`.

## Adding a New Lesson

1. Create `content/topics/<topic-id>/<slug>.md`.
2. Add the slug to `TOPIC_LESSON_ORDER[topic-id]` in `src/lib/topic-navigation.server.ts` at the desired position.
3. Add a summary to `LESSON_SUMMARIES[topic-id][slug]`.
4. If the lesson contains simulation placeholders, register them (see `docs/guides/simulations.md`).

## Lesson File Conventions

- Each lesson should include at least one heading: `# Lesson Title`
- Optional frontmatter block:

```md
---
title: Custom Lesson Title
author: Team
---
```

- If frontmatter `title` is missing, title falls back to first `#` heading, then slug.
- Landing pages use special slugs: `home`, `intro`, `introduction`, or `landingPage` (sorted before regular lessons).

## Supported Placeholders

- simulation: `[[simulation simulation-id]]`
- figure: `[[figure figure-id]]`
- code editor: `[[code-editor initial|expected|solution]]`

## Math And Markdown Notes

- Block math: `$$ ... $$` or `\\[ ... \\]`
- Inline math: `$ ... $` or `\\( ... \\)`
- Keep syntax simple because markdown rendering uses a custom transformer (`src/lib/markdown-to-html.ts`).

## Validation Workflow

After content edits:

1. Run `npm run check:simulations` if any simulation placeholder changed.
2. Run `npm test` for content gateway and domain behavior safety.
3. Open at least one modified lesson route in local dev (`/topics/<topic>/<slug>`).

## Do / Don't

### Do

- keep slugs stable once linked publicly
- keep placeholder ids exact and lowercase where possible
- use one topic directory per content domain id
- add summaries for every non-landing lesson in `LESSON_SUMMARIES`

### Don't

- invent new placeholder syntax without renderer update
- move topics without updating route and topic registries
- rely on undocumented frontmatter keys for runtime behavior
- forget to add lesson ordering when creating new lessons
