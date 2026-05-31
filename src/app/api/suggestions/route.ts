import { createClient } from "@/infra/supabase/server";
import { logger } from "@/infra/observability/logger";
import { AppError } from "@/shared/errors/app-error";
import { apiSuccess, withApiHandler } from "@/shared/errors/api-error";

export const runtime = "nodejs";

export const POST = withApiHandler("/api/suggestions", "POST", async (request, ctx) => {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    throw new AppError("BAD_REQUEST", "Invalid JSON body", 400);
  }
  const { suggestion, page, context } = (body ?? {}) as {
    suggestion?: unknown;
    page?: unknown;
    context?: unknown;
  };

  if (!suggestion || typeof suggestion !== "string") {
    throw new AppError("BAD_REQUEST", "Suggestion is required", 400);
  }

  const supabase = await createClient();
  if (!supabase) {
    throw new AppError("UNAUTHORIZED", "Authentication required", 401);
  }

  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    throw new AppError("UNAUTHORIZED", "Authentication required", 401);
  }

  const { error } = await supabase
    .from("suggestions")
    .insert({
      suggestion: suggestion.slice(0, 2000),
      page: typeof page === "string" ? page.slice(0, 200) : "/",
      user_id: user.id,
      status: "pending_injection_assessment",
      ...(context != null && context !== ""
        ? { context: String(context).slice(0, 100) }
        : {}),
    });

  if (error) {
    logger.error("Supabase insert error", { error });
    throw new AppError("INTERNAL_ERROR", "Failed to save suggestion", 500);
  }

  return apiSuccess(ctx, { success: true });
});
