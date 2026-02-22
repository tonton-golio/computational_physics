import { createClient } from "@/infra/supabase/server";
import { createClient as createAdminClient } from "@supabase/supabase-js";
import { logger } from "@/infra/observability/logger";
import { AppError } from "@/shared/errors/app-error";
import { apiSuccess, withApiHandler } from "@/shared/errors/api-error";

export const runtime = "nodejs";

export const DELETE = withApiHandler("/api/auth/delete-account", "DELETE", async (request, ctx) => {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    throw new AppError("UNAUTHORIZED", "Not authenticated", 401);
  }

  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!serviceRoleKey) {
    throw new AppError("INTERNAL_ERROR", "Service role key not configured", 500);
  }

  const admin = createAdminClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    serviceRoleKey,
    { auth: { autoRefreshToken: false, persistSession: false } }
  );

  const { error } = await admin.auth.admin.deleteUser(user.id);

  if (error) {
    logger.error("Delete user error", { error });
    throw new AppError("INTERNAL_ERROR", "Failed to delete account", 500);
  }

  return apiSuccess(ctx, { success: true });
});
