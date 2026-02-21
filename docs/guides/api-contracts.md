# API Contracts

## Purpose

Capture current HTTP contracts so contributors and agents can change endpoints safely.

## Global Conventions

- Runtime: Node.js route handlers (`runtime = "nodejs"` where declared).
- Validation: Zod at request boundary for structured inputs.
- Errors: `AppError` + `asErrorEnvelope()` shape.
- Correlation: `/api/content` sets `x-correlation-id` on every response.
- Logging: server-side API routes use `logger` from `src/infra/observability/logger.ts` (structured JSON).
- All `/api/*` routes have `X-Robots-Tag: noindex` (set in `vercel.json`).

## `GET /api/content`

Source: `src/app/api/content/route.ts`.

### Query Modes

- no query: `{ topics: TopicIndex[] }`
- `?topic=<topic-id>`: `{ topic: TopicDefinition, lessons: ContentDocument[] }`
- `?topic=<topic-id>&slug=<lesson-slug>`: `{ content: ContentDocument }`

Lessons are returned in `TOPIC_LESSON_ORDER` order (not alphabetical).

### Status Codes

- `200`: success
- `400`: invalid query parameters (`BAD_REQUEST`)
- `404`: unknown topic/lesson (`NOT_FOUND`)
- `500`: unexpected server failure (`INTERNAL_ERROR`)

### Cache Headers

- `Cache-Control: public, s-maxage=3600, stale-while-revalidate=86400`
- `revalidate = 3600`

### Error Envelope

```json
{
  "error": {
    "code": "BAD_REQUEST",
    "message": "Invalid query parameters.",
    "details": {}
  }
}
```

## `GET /api/content/export`

Source: `src/app/api/content/export/route.ts`.

- `?topic=<topic-id>`: returns `{ topic: TopicDefinition, lessons: ContentDocument[] }` — all ordered non-landing lessons for export.

## `GET /api/health`

Source: `src/app/api/health/route.ts`.

- `200` response:

```json
{
  "status": "ok",
  "service": "computational-physics",
  "timestamp": "ISO-8601"
}
```

## `GET /api/ready`

Source: `src/app/api/ready/route.ts`.

- `200` when topic index can load:

```json
{
  "status": "ready",
  "topics": 10,
  "timestamp": "ISO-8601"
}
```

- `503` when readiness check fails:

```json
{
  "status": "not_ready",
  "timestamp": "ISO-8601"
}
```

## `GET /api/auth/callback`

Source: `src/app/api/auth/callback/route.ts`.

- Exchanges OAuth code for Supabase session.
- Redirects to `?next=` parameter or falls back to `/login?error=...` on failure.

## `DELETE /api/auth/delete-account`

Source: `src/app/api/auth/delete-account/route.ts`.

- Requires authenticated session.
- Uses Supabase admin client (service role key) to delete the user.
- `200`: `{ success: true }`
- `401`: not authenticated
- `500`: deletion failed

## `POST /api/suggestions`

Source: `src/app/api/suggestions/route.ts`.

- Requires authenticated session.
- Body: `{ suggestion: string, page?: string }`
- Inserts into Supabase `suggestions` table.
- `200`: `{ success: true }`
- `400`: missing or invalid suggestion
- `401`: not authenticated
- `500`: database insert failed

## `POST /api/analytics/web-vitals`

Source: `src/app/api/analytics/web-vitals/route.ts`.

- Body accepts optional fields: `id`, `name`, `value`, `rating`, `path`, `navigationType`, `timestamp`
- `202` on valid payload: `{ "ok": true }`
- `400` on invalid payload: `{ "ok": false }`

## Contract Change Checklist

- Update this file for any endpoint shape/status/header changes.
- Preserve backward-compatible fields unless coordinated migration.
- Add/update tests for changed behavior and error cases.
