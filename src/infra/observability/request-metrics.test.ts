import { describe, expect, it, vi } from "vitest";
import { correlationIdFrom, logRequestMetric } from "./request-metrics";

describe("request-metrics", () => {
  describe("correlationIdFrom", () => {
    it("returns existing correlation ID from headers", () => {
      const request = new Request("https://example.com", {
        headers: { "x-correlation-id": "test-id-123" },
      });
      expect(correlationIdFrom(request)).toBe("test-id-123");
    });

    it("generates UUID when header is absent", () => {
      const request = new Request("https://example.com");
      const id = correlationIdFrom(request);
      expect(id).toBeTruthy();
      // UUID v4 format
      expect(id).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
      );
    });

    it("generates unique IDs on each call without header", () => {
      const req1 = new Request("https://example.com");
      const req2 = new Request("https://example.com");
      expect(correlationIdFrom(req1)).not.toBe(correlationIdFrom(req2));
    });
  });

  describe("logRequestMetric", () => {
    it("logs structured metric via console.info", () => {
      const spy = vi.spyOn(console, "info").mockImplementation(() => {});
      logRequestMetric({
        route: "/api/test",
        method: "GET",
        status: 200,
        durationMs: 42.7,
        correlationId: "abc",
      });
      expect(spy).toHaveBeenCalledOnce();
      const logged = JSON.parse(spy.mock.calls[0][0] as string);
      expect(logged.message).toBe("request.metric");
      expect(logged.context.route).toBe("/api/test");
      expect(logged.context.method).toBe("GET");
      expect(logged.context.status).toBe(200);
      expect(logged.context.durationMs).toBe(43); // rounded
      expect(logged.context.correlationId).toBe("abc");
      spy.mockRestore();
    });
  });
});
