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
  return request.headers.get("x-correlation-id") ?? crypto.randomUUID();
}
