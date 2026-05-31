import { describe, expect, it } from "vitest";
import { apiSuccess, createApiContext, withApiHandler } from "./api-error";
import { AppError } from "./app-error";

describe("withApiHandler", () => {
  it("maps a thrown AppError to its status, code, and preserves the correlation id", async () => {
    const route = withApiHandler("/x", "GET", async () => {
      throw new AppError("NOT_FOUND", "x", 404);
    });

    const request = new Request("http://localhost/x", {
      headers: { "x-correlation-id": "abc" },
    });
    const res = await route(request);

    expect(res.status).toBe(404);
    const body = await res.json();
    expect(body.error.code).toBe("NOT_FOUND");
    expect(res.headers.get("x-correlation-id")).toBe("abc");
  });

  it("maps an unknown thrown Error to a redacted 500 envelope with no details", async () => {
    const route = withApiHandler("/x", "GET", async () => {
      throw new Error("boom");
    });

    const res = await route(new Request("http://localhost/x"));

    expect(res.status).toBe(500);
    const body = await res.json();
    expect(body.error.code).toBe("INTERNAL_ERROR");
    expect(body.error.details).toBeUndefined();
    expect("details" in body.error).toBe(false);
  });
});

describe("apiSuccess", () => {
  it("merges extra headers and always sets a correlation id", async () => {
    const request = new Request("http://localhost/x");
    const ctx = createApiContext("/x", "GET", request);

    const res = apiSuccess(ctx, { ok: true }, 200, { "Cache-Control": "x" });

    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe("x");
    expect(res.headers.get("x-correlation-id")).toBeTruthy();
    const body = await res.json();
    expect(body.ok).toBe(true);
  });
});
