import { z } from "zod";
import { getTopicLessons } from "@/features/content/content-gateway";
import { getLessonSummary } from "@/features/content/topic-lessons";
import { TOPICS } from "@/lib/topic-config";
import { AppError } from "@/shared/errors/app-error";
import { apiSuccess, withApiHandler } from "@/shared/errors/api-error";

export const runtime = "nodejs";

const querySchema = z.object({
  topic: z.string().min(1),
});

export const GET = withApiHandler("/api/content/export", "GET", async (request, ctx) => {
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

  const { lessons: docs } = await getTopicLessons(topic);

  const lessons = docs.map((doc) => {
    const summary = getLessonSummary(topic, doc.slug);
    return { slug: doc.slug, title: doc.title, content: doc.content, ...(summary ? { summary } : {}) };
  });

  return apiSuccess(ctx, { topic, lessons });
});
