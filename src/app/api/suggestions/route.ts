import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";
import { logger } from "@/infra/observability/logger";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { suggestion, page, context } = body;

    if (!suggestion || typeof suggestion !== "string") {
      return NextResponse.json({ error: "Suggestion is required" }, { status: 400 });
    }

    const supabase = await createClient();

    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return NextResponse.json({ error: "Authentication required" }, { status: 401 });
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
      return NextResponse.json({ error: "Failed to save suggestion" }, { status: 500 });
    }

    return NextResponse.json({ success: true });
  } catch {
    return NextResponse.json({ error: "Failed to save suggestion" }, { status: 500 });
  }
}
