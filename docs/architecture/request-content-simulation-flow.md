# Request, Content, And Simulation Flow

## Purpose

Describe the end-to-end runtime path from request to rendered lesson, including fallback and error behavior.

## Main Flow

```mermaid
flowchart TD
  userRequest[User Requests Lesson Route] --> lessonPage[/topics/[topic]/[slug]/page.tsx]
  lessonPage --> topicRoute[resolveTopicRoute]
  topicRoute --> lessonRead[readLessonDocument]
  lessonRead --> markdownRender[MarkdownContent]
  markdownRender --> placeholderScan[Extract Placeholders]
  placeholderScan --> simulationHost[SimulationHost]
  simulationHost --> manifestResolve[resolveSimulationDefinition]
  manifestResolve --> topicRegistry[Topic *Simulations.tsx Registry]
  topicRegistry --> simulationComponent[Simulation Component]
  simulationComponent --> finalRender[Page Render Complete]
```

## Content Pipeline

```
content/topics/<topicId>/<slug>.md
        |
        v  (fs.readFileSync + frontmatter parse + Zod validate)
src/infra/content/file-content-repository.ts
  -> listLessonSlugs(topicId): string[]
  -> readLessonDocument(topicId, slug): ContentDocument | null
        |
        v
src/lib/topic-navigation.server.ts
  -> getOrderedLessonSlugs(topicId): string[]  (applies TOPIC_LESSON_ORDER)
        |
        v
src/features/content/content-gateway.ts
  -> getTopicIndexes()  [Next.js unstable_cache, 3600s revalidate]
  -> getTopicLessons(topicId)  [ordered by TOPIC_LESSON_ORDER]
  -> getTopicLesson(topicId, slug)
        |
        v  (for API)              v  (for pages, direct call)
src/app/api/content/route.ts    src/app/topics/[topic]/[slug]/page.tsx
```

## Content API Flow (`/api/content`)

Source: `src/app/api/content/route.ts`.

- Query parameters:
  - no params: returns all topic indexes
  - `topic`: returns topic with ordered lessons
  - `topic` + `slug`: returns single lesson document
- Validation uses Zod and rejects invalid params with `BAD_REQUEST` (400).
- All responses set `x-correlation-id`.
- Cache-Control: `public, s-maxage=3600, stale-while-revalidate=86400`
- Errors are returned with `asErrorEnvelope()`.

## Lesson Page Flow (`/topics/[topic]/[slug]`)

Source: `src/app/topics/[topic]/[slug]/page.tsx`.

- `generateStaticParams()` precomputes lesson routes from `TOPIC_ROUTES` and filesystem lesson slugs.
- Route slug resolves to content topic via `resolveTopicRoute()`.
- Lesson content is read from `content/topics/<topic>/<slug>.md|.txt`.
- Missing route or lesson triggers `notFound()`.
- Content body is rendered by client component `MarkdownContent`.
- Sidebar shows ordered lessons from `getLessonsForTopic()`.

## Placeholder Processing

Source: `src/lib/markdown-to-html.ts` and `src/components/content/MarkdownContent.tsx`.

- Placeholder formats:
  - `[[simulation <id>]]` — simulation component
  - `[[figure <id>]]` — image with caption
  - `[[code-editor <payload>]]` — interactive code editor
- Processing:
  1. `markdownToHtml()` replaces placeholders with `%%PLACEHOLDER_N%%` markers
  2. Renders LaTeX, markdown formatting, code blocks
  3. `MarkdownContent.tsx` splits on markers, renders HTML chunks as `dangerouslySetInnerHTML`
  4. Placeholder chunks become React components: `SimulationHost`, `Image`, `CodeEditor`, `CodeToggleBlock`
- Unknown placeholder type renders a visible error block.

## Simulation Resolution And Fallback

Sources: `src/features/simulation/simulation-manifest.ts`, `src/features/simulation/SimulationHost.tsx`.

- `SimulationHost` defers expensive loads until near/in viewport (IntersectionObserver).
- Definitions lazily resolved by id across 11 topic registries.
- Fast path: `ONLINE_REINFORCEMENT_IDS` set avoids scanning all registries for known RL simulation ids.
- Registry load failures are isolated (one bad registry does not break all simulations).
- If simulation id is not found, host falls back to `InteractiveGraph`.
- If both simulation and fallback fail, host renders an error placeholder.
- Results are cached: per-registry import cache + per-id `SimulationDefinition` cache with in-flight deduplication.

## Reliability Signals

- `/api/health`: simple liveness JSON (`status: ok`).
- `/api/ready`: content index check; returns `503` when topic index load fails.
- `/api/analytics/web-vitals`: accepts metric payloads, responds `202` on valid body.
- `logRequestMetric()` emits request timing + status + correlation id.

## Actionable Checklist

- For content rendering regressions:
  - verify `resolveTopicRoute()` mapping in `src/lib/topic-navigation.ts`
  - verify file exists under `content/topics/<topic-id>/`
  - verify markdown placeholder syntax
- For simulation regressions:
  - run `npm run check:simulations`
  - confirm simulation id exists in topic `*Simulations.tsx` registry
  - verify `simulation-manifest.ts` includes topic registry group
- For ordering issues:
  - check `TOPIC_LESSON_ORDER` in `src/lib/topic-navigation.server.ts`
