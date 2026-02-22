# Versioning

The landing page displays a version string in the format `VXX.YY.ZZZ`.

| Segment | Meaning | Who bumps it |
|---------|---------|--------------|
| `XX` | Major — only incremented by Anton's explicit request | Anton |
| `YY` | Minor — new topics added to the platform | Any contributor |
| `ZZZ` | Patch — content updates, fixes, and improvements | Any contributor |

Current version: **V02.01.002**

## Pre-commit checklist

Before every commit, update the following:

### 1. Bump the version number

The version string is hardcoded in `src/app/page.tsx` on the landing page header line:

```tsx
<p className="...">koala brain :: V02.01.002</p>
```

Bump the appropriate segment based on your change (see table above) and update both:
- The version in `src/app/page.tsx`
- The "Current version" line in this file (`docs/versioning.md`)

### 2. Verify topic and subtopic counts

The landing page dynamically computes topic and subtopic counts from `TOPIC_ROUTES` (in `src/lib/topic-navigation.ts`) and `getLessonsForTopic` (in `src/lib/topic-navigation.server.ts`). If you added or removed topics or lessons:

- Ensure new topics are registered in `TOPIC_ROUTES` in `src/lib/topic-navigation.ts`
- Ensure lesson ordering is set in `TOPIC_LESSON_ORDER` in `src/lib/topic-navigation.server.ts`
- Run `npm run dev` and confirm the landing page shows the correct counts

### 3. Regenerate the topic point cloud

If you added, removed, or renamed topics or lessons, re-run the point cloud generation so the topics visualization stays in sync:

```bash
npm run build:points
```

This requires `uv` and `OPENAI_API_KEY` in `.env`. The script caches embeddings, so only new or changed lessons trigger API calls. The output (`public/data/topic-points.json`) must be committed alongside your content changes.
