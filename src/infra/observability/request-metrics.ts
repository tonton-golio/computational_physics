import { logger } from "./logger";

export interface RequestMetricInput {
  route: string;
  method: string;
  status: number;
  durationMs: number;
  correlationId: string;
}

export function logRequestMetric(input: RequestMetricInput): void {
  logger.info("request.metric", {
    route: input.route,
    method: input.method,
    status: input.status,
    durationMs: Math.round(input.durationMs),
    correlationId: input.correlationId,
  });
}

export function correlationIdFrom(request: Request): string {
  const provided = request.headers.get("x-correlation-id");
  if (provided) {
    return provided.replace(/[^A-Za-z0-9._-]/g, "").slice(0, 128);
  }
  return crypto.randomUUID();
}
