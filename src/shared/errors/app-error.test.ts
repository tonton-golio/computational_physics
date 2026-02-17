import { describe, expect, it } from "vitest";
import { AppError, asErrorEnvelope } from "./app-error";

describe("AppError", () => {
  it("creates typed error envelope", () => {
    const envelope = asErrorEnvelope(new AppError("NOT_FOUND", "Missing", 404, { id: "x" }));
    expect(envelope.error.code).toBe("NOT_FOUND");
    expect(envelope.error.message).toBe("Missing");
  });

  it("maps unknown errors to internal envelope", () => {
    const envelope = asErrorEnvelope(new Error("boom"));
    expect(envelope.error.code).toBe("INTERNAL_ERROR");
  });
});
