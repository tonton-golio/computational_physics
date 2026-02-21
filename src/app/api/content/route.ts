import { z } from "zod";
import { NextResponse } from "next/server";
import { getTopicIndexes, getTopicLesson, getTopicLessons } from "@/features/content/content-gateway";
import { logRequestMetric, correlationIdFrom } from "@/infra/observability/request-metrics";
import { AppError, asErrorEnvelope } from "@/shared/errors/app-error";

export const runtime = "nodejs";
export const revalidate = 3600;

const CACHE_HEADERS = {
  "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
};

const querySchema = z.object({
  topic: z.string().optional(),
  slug: z.string().optional(),
});

export async function GET(request: Request) {
  const startedAt = performance.now();
  const correlationId = correlationIdFrom(request);

  try {
    const { searchParams } = new URL(request.url);
    const parsed = querySchema.safeParse({
      topic: searchParams.get("topic") ?? undefined,
      slug: searchParams.get("slug") ?? undefined,
    });

    if (!parsed.success) {
      throw new AppError("BAD_REQUEST", "Invalid query parameters.", 400, parsed.error.issues);
    }

    const { topic, slug } = parsed.data;

    if (!topic) {
      const topics = await getTopicIndexes();
      const response = NextResponse.json({ topics }, { headers: CACHE_HEADERS });
      response.headers.set("x-correlation-id", correlationId);
      logRequestMetric({
        route: "/api/content",
        method: "GET",
        status: 200,
        durationMs: performance.now() - startedAt,
        correlationId,
      });
      return response;
    }

    if (topic && slug) {
      const content = await getTopicLesson(topic, slug);
      if (!content) {
        throw new AppError("NOT_FOUND", "Content not found.", 404, { topic, slug });
      }
      const response = NextResponse.json({ content }, { headers: CACHE_HEADERS });
      response.headers.set("x-correlation-id", correlationId);
      logRequestMetric({
        route: "/api/content",
        method: "GET",
        status: 200,
        durationMs: performance.now() - startedAt,
        correlationId,
      });
      return response;
    }

    const lessonSet = await getTopicLessons(topic);
    const response = NextResponse.json(lessonSet, { headers: CACHE_HEADERS });
    response.headers.set("x-correlation-id", correlationId);
    logRequestMetric({
      route: "/api/content",
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
      route: "/api/content",
      method: "GET",
      status,
      durationMs: performance.now() - startedAt,
      correlationId,
    });
    return response;
  }
}
