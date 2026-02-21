import { test, expect } from "@playwright/test";

test.describe("API health endpoints", () => {
  test("GET /api/health returns ok status", async ({ request }) => {
    const response = await request.get("/api/health");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.status).toBe("ok");
    expect(body.service).toBe("computational-physics");
    expect(body.timestamp).toBeTruthy();
  });

  test("GET /api/ready returns ready with topic count", async ({ request }) => {
    const response = await request.get("/api/ready");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.status).toBe("ready");
    expect(body.topics).toBeGreaterThanOrEqual(10);
  });
});
