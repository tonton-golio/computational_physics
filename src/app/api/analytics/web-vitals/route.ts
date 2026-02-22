import { z } from "zod";
import { logger } from "@/infra/observability/logger";
import { AppError } from "@/shared/errors/app-error";
import { apiSuccess, withApiHandler } from "@/shared/errors/api-error";

export const runtime = "nodejs";

const metricSchema = z.object({
  id: z.string().optional(),
  name: z.string().optional(),
  value: z.number().optional(),
  rating: z.string().optional(),
  path: z.string().optional(),
  navigationType: z.string().optional(),
  timestamp: z.number().optional(),
});

export const POST = withApiHandler("/api/analytics/web-vitals", "POST", async (request, ctx) => {
  try {
    const payload = metricSchema.parse(await request.json());

    if (process.env.NODE_ENV !== "production" || process.env.LOG_WEB_VITALS === "1") {
      logger.info("web-vitals.metric", {
        id: payload.id,
        name: payload.name,
        value: payload.value,
        rating: payload.rating,
        path: payload.path,
        navigationType: payload.navigationType,
        timestamp: payload.timestamp,
      });
    }

    return apiSuccess(ctx, { ok: true }, 202);
  } catch (err) {
    if (err instanceof z.ZodError) {
      throw new AppError("BAD_REQUEST", "Invalid metric payload", 400, err.issues);
    }
    throw err;
  }
});
