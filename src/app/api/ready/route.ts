import { NextResponse } from "next/server";
import { getTopicIndexes } from "@/features/content/content-gateway";

export const runtime = "nodejs";

export async function GET() {
  try {
    const topics = await getTopicIndexes();
    return NextResponse.json(
      {
        status: "ready",
        topics: topics.length,
        timestamp: new Date().toISOString(),
      },
      { status: 200 }
    );
  } catch {
    return NextResponse.json(
      {
        status: "not_ready",
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
}
