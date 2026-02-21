import { describe, expect, it } from "vitest";
import { AppError, asErrorEnvelope } from "./app-error";

describe("AppError", () => {
  describe("constructor", () => {
    it("sets name to AppError", () => {
      const err = new AppError("NOT_FOUND", "gone", 404);
      expect(err.name).toBe("AppError");
    });

    it("extends Error", () => {
      const err = new AppError("BAD_REQUEST", "bad", 400);
      expect(err).toBeInstanceOf(Error);
      expect(err).toBeInstanceOf(AppError);
    });

    it("defaults status to 500", () => {
      const err = new AppError("INTERNAL_ERROR", "oops");
      expect(err.status).toBe(500);
    });

    it("stores code, message, status, and details", () => {
      const err = new AppError("CONTENT_READ_ERROR", "read failed", 503, { path: "/x" });
      expect(err.code).toBe("CONTENT_READ_ERROR");
      expect(err.message).toBe("read failed");
      expect(err.status).toBe(503);
      expect(err.details).toEqual({ path: "/x" });
    });

    it("details is undefined when not provided", () => {
      const err = new AppError("NOT_FOUND", "nope", 404);
      expect(err.details).toBeUndefined();
    });
  });

  describe("asErrorEnvelope", () => {
    it("creates typed error envelope from AppError", () => {
      const envelope = asErrorEnvelope(new AppError("NOT_FOUND", "Missing", 404, { id: "x" }));
      expect(envelope.error.code).toBe("NOT_FOUND");
      expect(envelope.error.message).toBe("Missing");
      expect(envelope.error.details).toEqual({ id: "x" });
    });

    it("maps unknown errors to internal envelope", () => {
      const envelope = asErrorEnvelope(new Error("boom"));
      expect(envelope.error.code).toBe("INTERNAL_ERROR");
      expect(envelope.error.message).toBe("An unexpected error occurred.");
      expect(envelope.error.details).toBeUndefined();
    });

    it("maps string throw to internal envelope", () => {
      const envelope = asErrorEnvelope("string error");
      expect(envelope.error.code).toBe("INTERNAL_ERROR");
    });

    it("maps null to internal envelope", () => {
      const envelope = asErrorEnvelope(null);
      expect(envelope.error.code).toBe("INTERNAL_ERROR");
    });

    it("preserves all AppError codes", () => {
      const codes = [
        "BAD_REQUEST",
        "NOT_FOUND",
        "CONTENT_READ_ERROR",
        "CONTENT_PARSE_ERROR",
        "INTERNAL_ERROR",
      ] as const;
      for (const code of codes) {
        const envelope = asErrorEnvelope(new AppError(code, `msg-${code}`));
        expect(envelope.error.code).toBe(code);
      }
    });
  });
});
