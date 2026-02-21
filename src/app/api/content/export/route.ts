import { z } from "zod";
import { NextResponse } from "next/server";
import { readLessonDocument } from "@/infra/content/file-content-repository";
import { getOrderedLessonSlugs, getLessonSummary } from "@/lib/topic-navigation.server";
import { TOPICS } from "@/lib/content";
import { logRequestMetric, correlationIdFrom } from "@/infra/observability/request-metrics";
import { AppError, asErrorEnvelope } from "@/shared/errors/app-error";

export const runtime = "nodejs";

const querySchema = z.object({
  topic: z.string().min(1),
});

export async function GET(request: Request) {
  const startedAt = performance.now();
  const correlationId = correlationIdFrom(request);

  try {
    const { searchParams } = new URL(request.url);
    const parsed = querySchema.safeParse({
      topic: searchParams.get("topic") ?? undefined,
    });

    if (!parsed.success) {
      throw new AppError("BAD_REQUEST", "Missing or invalid 'topic' query parameter.", 400, parsed.error.issues);
    }

    const { topic } = parsed.data;

    if (!(topic in TOPICS)) {
      throw new AppError("NOT_FOUND", `Topic "${topic}" not found.`, 404, { topic });
    }

    const slugs = getOrderedLessonSlugs(topic);

    const lessons = slugs
      .map((slug) => {
        const doc = readLessonDocument(topic, slug);
        if (!doc) return null;
        const summary = getLessonSummary(topic, slug);
        return { slug: doc.slug, title: doc.title, content: doc.content, ...(summary ? { summary } : {}) };
      })
      .filter((l): l is { slug: string; title: string; content: string; summary?: string } => l !== null);

    const response = NextResponse.json({ topic, lessons });
    response.headers.set("x-correlation-id", correlationId);
    logRequestMetric({
      route: "/api/content/export",
      method: "GET",
      status: 200,
      durationMs: performance.now() - startedAt,
      correlationId,
    });
    return response;
  } catch (error) {
    const status = error instanceof AppError ? error.status : 500;
    const response = NextResponse.json(asErrorEnvelope(error), { status });
    response.headers.set("x-correlation-id", correlationId);
    logRequestMetric({
      route: "/api/content/export",
      method: "GET",
      status,
      durationMs: performance.now() - startedAt,
      correlationId,
    });
    return response;
  }
}
