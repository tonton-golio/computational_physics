import { createClient } from "@/infra/supabase/server";
import { logger } from "@/infra/observability/logger";
import { AppError } from "@/shared/errors/app-error";
import { apiSuccess, withApiHandler } from "@/shared/errors/api-error";

export const runtime = "nodejs";

export const POST = withApiHandler("/api/suggestions", "POST", async (request, ctx) => {
  const body = await request.json();
  const { suggestion, page, context } = body;

  if (!suggestion || typeof suggestion !== "string") {
    throw new AppError("BAD_REQUEST", "Suggestion is required", 400);
  }

  const supabase = await createClient();

  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    throw new AppError("UNAUTHORIZED", "Authentication required", 401);
  }

  const { error } = await supabase
    .from("suggestions")
    .insert({
      suggestion: suggestion.slice(0, 2000),
      page: page || "/",
      user_id: user.id,
      ...(context && { context: String(context).slice(0, 100) }),
    });

  if (error) {
    logger.error("Supabase insert error", { error });
    throw new AppError("INTERNAL_ERROR", "Failed to save suggestion", 500);
  }

  return apiSuccess(ctx, { success: true });
});
