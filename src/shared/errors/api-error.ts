import { NextResponse } from "next/server";
import { AppError, asErrorEnvelope } from "./app-error";
import { logRequestMetric, correlationIdFrom } from "@/infra/observability/request-metrics";

export interface ApiContext {
  route: string;
  method: string;
  request: Request;
  startedAt: number;
}

export function createApiContext(route: string, method: string, request: Request): ApiContext {
  return { route, method, request, startedAt: performance.now() };
}

export function apiSuccess(ctx: ApiContext, data: unknown, status = 200, headers?: Record<string, string>): NextResponse {
  const correlationId = correlationIdFrom(ctx.request);
  const response = NextResponse.json(data, { status });
  response.headers.set("x-correlation-id", correlationId);
  if (headers) {
    for (const [key, value] of Object.entries(headers)) {
      response.headers.set(key, value);
    }
  }
  logRequestMetric({
    route: ctx.route,
    method: ctx.method,
    status,
    durationMs: performance.now() - ctx.startedAt,
    correlationId,
  });
  return response;
}

export function apiError(ctx: ApiContext, error: unknown): NextResponse {
  const correlationId = correlationIdFrom(ctx.request);
  const status = error instanceof AppError ? error.status : 500;
  const response = NextResponse.json(asErrorEnvelope(error), { status });
  response.headers.set("x-correlation-id", correlationId);
  logRequestMetric({
    route: ctx.route,
    method: ctx.method,
    status,
    durationMs: performance.now() - ctx.startedAt,
    correlationId,
  });
  return response;
}

export function withApiHandler(
  route: string,
  method: string,
  handler: (request: Request, ctx: ApiContext) => Promise<NextResponse>,
): (request: Request) => Promise<NextResponse> {
  return async (request: Request) => {
    const ctx = createApiContext(route, method, request);
    try {
      return await handler(request, ctx);
    } catch (err) {
      return apiError(ctx, err);
    }
  };
}
