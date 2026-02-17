import { NextResponse } from "next/server";
import { z } from "zod";
import { logger } from "@/infra/observability/logger";

const metricSchema = z.object({
  id: z.string().optional(),
  name: z.string().optional(),
  value: z.number().optional(),
  rating: z.string().optional(),
  path: z.string().optional(),
  navigationType: z.string().optional(),
  timestamp: z.number().optional(),
});

export async function POST(request: Request) {
  try {
    const payload = metricSchema.parse(await request.json());

    // Keep this endpoint lightweight so it remains safe under high traffic.
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

    return NextResponse.json({ ok: true }, { status: 202 });
  } catch {
    return NextResponse.json({ ok: false }, { status: 400 });
  }
}
