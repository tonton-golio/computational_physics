import { z } from "zod";
import { getTopicIndexes, getTopicLesson, getTopicLessons } from "@/features/content/content-gateway";
import { AppError } from "@/shared/errors/app-error";
import { apiSuccess, withApiHandler } from "@/shared/errors/api-error";

export const runtime = "nodejs";
export const revalidate = 3600;

const CACHE_HEADERS = {
  "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
};

const querySchema = z.object({
  topic: z.string().optional(),
  slug: z.string().optional(),
});

export const GET = withApiHandler("/api/content", "GET", async (request, ctx) => {
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
    return apiSuccess(ctx, { topics }, 200, CACHE_HEADERS);
  }

  if (topic && slug) {
    const content = await getTopicLesson(topic, slug);
    if (!content) {
      throw new AppError("NOT_FOUND", "Content not found.", 404, { topic, slug });
    }
    return apiSuccess(ctx, { content }, 200, CACHE_HEADERS);
  }

  const lessonSet = await getTopicLessons(topic);
  return apiSuccess(ctx, lessonSet, 200, CACHE_HEADERS);
});
