import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json(
    {
      status: "ok",
      service: "computational-physics",
      timestamp: new Date().toISOString(),
    },
    { status: 200 }
  );
}
